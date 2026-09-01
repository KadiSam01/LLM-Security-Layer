from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import httpx, os, json, asyncio, re

app = FastAPI()

# === Config ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama2:7b")

# === Memory Stores ===
conversation_history = {}       # per-session messages
conversation_summary = {}       # per-session summary
global_memory = []              # global conversation summaries (cross-session)
knowledge_base = set()          # facts or statements extracted from all chats

# === Utility: Extract meaningful facts from text ===
def extract_facts(text: str):
    """
    Extract general knowledge-like facts from user messages.
    Example: "I study at Xavier University" → stores that statement verbatim.
    """
    # Keep sentences that include verbs or key info
    facts = re.split(r'(?<=[.!?])\s+', text.strip())
    for f in facts:
        if len(f.split()) > 3 and not f.lower().startswith(("assistant", "user", "system")):
            knowledge_base.add(f.strip())

# === Summarization ===
async def summarize_conversation(session_id: str):
    """Summarize and merge memory globally."""
    history = conversation_history.get(session_id, [])
    if len(history) < 4:
        return

    recent_msgs = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history[-8:]])

    # Build base prompt for summarization
    facts_text = "\n".join(sorted(list(knowledge_base))[-20:]) if knowledge_base else ""
    summary_prompt = (
        "You are summarizing a user conversation. Write a compact factual summary that captures all key details, "
        "preferences, facts, and important events the user mentioned. "
        "Preserve meaning exactly and don't add new information.\n\n"
        f"Existing global knowledge:\n{facts_text}\n\n"
        f"Conversation:\n{recent_msgs}\n\nSummary:"
    )

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": DEFAULT_MODEL, "prompt": summary_prompt, "stream": False},
            )
            resp.raise_for_status()
            summary = resp.json().get("response", "").strip()

            # Save and merge globally
            conversation_summary[session_id] = summary
            conversation_history[session_id] = history[-10:]
            if summary not in [m["summary"] for m in global_memory]:
                global_memory.append({"session_id": session_id, "summary": summary})
                if len(global_memory) > 30:
                    global_memory.pop(0)

            # Also extract facts for next sessions
            extract_facts(summary)

        except Exception as e:
            print("⚠️ Summarization failed:", e)

# === Topic Change Detection ===
def detect_topic_change(session_id: str, message: str) -> bool:
    """Detect when the topic is unrelated."""
    history = conversation_history.get(session_id, [])
    if not history:
        return False

    last_msg = history[-1]["content"].lower()
    new_msg = message.lower()
    overlap = len(set(new_msg.split()) & set(last_msg.split()))
    return overlap < 2 and len(history) > 3

# === Prompt Builder ===
def build_prompt(session_id: str, user_message: str) -> str:
    """Construct a global context-aware prompt."""
    history = conversation_history.get(session_id, [])
    summary = conversation_summary.get(session_id, "")

    prompt = (
        "You are an intelligent AI assistant that remembers everything important from all past user interactions. "
        "Use both per-session memory and global memory to stay consistent. "
        "Do not invent details — only recall what the user has truly said.\n\n"
    )

    # Include global summaries
    if global_memory:
        prompt += "=== Global Memory (Summaries from all sessions) ===\n"
        for mem in global_memory[-10:]:
            prompt += f"Session {mem['session_id'][:8]}: {mem['summary']}\n"
        prompt += "\n"

    # Include extracted facts
    if knowledge_base:
        prompt += "=== User Knowledge Base ===\n"
        prompt += "\n".join(sorted(list(knowledge_base))[-25:]) + "\n\n"

    # Include current session summary
    if summary:
        prompt += f"=== This Session Summary ===\n{summary}\n\n"

    # Include last few exchanges
    for msg in history[-10:]:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

    prompt += f"User: {user_message}\nAssistant:"
    return prompt

# === File Upload Handling ===
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload")
async def upload_file(session_id: str = Form("anonymous"), file: UploadFile = File(...)):
    """Handles file uploads and links them to the user memory."""
    try:
        filename = f"{session_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(await file.read())
        msg = f"[User uploaded file: {file.filename}]"
        conversation_history.setdefault(session_id, []).append(
            {"role": "system", "content": msg, "file_path": filepath}
        )
        global_memory.append({"session_id": session_id, "summary": f"User uploaded file '{file.filename}'."})
        extract_facts(msg)
        return JSONResponse({"success": True, "filename": file.filename, "path": filepath})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# === Chat Endpoint ===
@app.post("/api/chat")
async def chat(request: Request):
    """Chat endpoint with streaming and persistent memory."""
    data = await request.json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "anonymous")
    model = data.get("model") or DEFAULT_MODEL
    stream = bool(data.get("stream", True))
    file_path = data.get("file_path")

    # Include uploaded file contents if present
    file_context = ""
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", errors="ignore") as f:
                file_context = f.read(2000)
        except Exception as e:
            print("⚠️ File read failed:", e)
    if file_context:
        user_message += f"\n\n---\nFile content:\n{file_context}\n---"

    # Maintain session
    conversation_history.setdefault(session_id, [])
    if detect_topic_change(session_id, user_message):
        print(f"🔄 Topic shift detected for {session_id}")
        conversation_summary[session_id] = conversation_summary.get(session_id, "")
        conversation_history[session_id] = []

    # Append user message and extract facts
    conversation_history[session_id].append({"role": "user", "content": user_message})
    extract_facts(user_message)
    asyncio.create_task(summarize_conversation(session_id))

    full_prompt = build_prompt(session_id, user_message)

    # === Streaming Mode ===
    async def generate_stream():
        response_text = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "prompt": full_prompt, "stream": True},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except:
                        yield f"data: {json.dumps({'delta': line})}\n\n"
                        continue
                    if payload.get("response"):
                        chunk = payload["response"]
                        response_text += chunk
                        yield f"data: {json.dumps({'delta': chunk})}\n\n"
                    if payload.get("done"):
                        break
        conversation_history[session_id].append({"role": "assistant", "content": response_text.strip()})
        extract_facts(response_text)
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
        )
    r.raise_for_status()
    text = r.json().get("response", "").strip()
    conversation_history[session_id].append({"role": "assistant", "content": text})
    extract_facts(text)
    return JSONResponse({"text": text, "sources": []})
