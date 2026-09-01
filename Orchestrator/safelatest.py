from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, Request, Form, File, UploadFile
import httpx, os, json, asyncio, re

app = FastAPI()

# Internal service URLs / defaults
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DEFAULT_MODEL   = os.getenv("DEFAULT_MODEL", "llama2:7b")

# 🧠 In-memory storage for conversations { session_id: [ {"role": "user"/"assistant", "content": "..."} ] }
conversation_history = {}
conversation_summary = {}  # stores condensed summaries per session


async def summarize_conversation(session_id: str):
    """Summarize the last 10 exchanges to reduce context size"""
    history = conversation_history.get(session_id, [])
    if not history:
        return

    # Only summarize if conversation is long enough
    if len(history) > 10:
        last_messages = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history[-10:]])
        summary_prompt = (
            "Summarize the following conversation in 3 concise sentences focusing only on key facts:\n\n"
            f"{last_messages}\n\nSummary:"
        )

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": DEFAULT_MODEL, "prompt": summary_prompt, "stream": False},
                )
                resp.raise_for_status()
                out = resp.json()
                summary = out.get("response", "").strip()
                # Save summary to session
                conversation_summary[session_id] = summary
                # Keep only last 10 messages to prevent overflow
                conversation_history[session_id] = history[-10:]
            except Exception as e:
                print("Summarization failed:", e)


def detect_topic_change(session_id: str, user_message: str) -> bool:
    """Detect if the user’s new message is unrelated to the prior topic"""
    history = conversation_history.get(session_id, [])
    if not history:
        return False

    # Get last assistant or user message
    last_msg = history[-1]["content"].lower()

    # Simple topic-shift heuristics
    # If new question contains math, code, or generic unrelated phrasing
    if re.search(r"\b(\d+\s*[\+\-\*/]\s*\d+|what is|who is|define|calculate|sum|add)\b", user_message.lower()):
        # Check if the last topic was not about math/calc
        if not re.search(r"\b(\d+|math|calculate|sum|add|number|compute)\b", last_msg):
            return True

    # Otherwise, compute simple keyword overlap
    overlap = len(set(user_message.lower().split()) & set(last_msg.split()))
    return overlap < 2  # fewer than 2 overlapping words means topic likely changed


def build_prompt(session_id: str, user_message: str) -> str:
    """Builds a structured prompt using past conversation without repetition"""
    history = conversation_history.get(session_id, [])
    summary = conversation_summary.get(session_id, "")

    formatted = (
        "You are a helpful AI assistant. Use the conversation history below to stay consistent, "
        "but do NOT repeat or restate previous messages. Respond naturally and concisely.\n\n"
    )

    # Include the summarized context first if it exists
    if summary:
        formatted += f"Summary of previous context: {summary}\n\n"

    # Include the last few exchanges for detailed continuity
    for msg in history[-10:]:
        role = msg["role"].capitalize()
        content = msg["content"]
        formatted += f"{role}: {content}\n"

    formatted += f"User: {user_message}\nAssistant:"
    return formatted


UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload")
async def upload_file(session_id: str = Form("anonymous"), file: UploadFile = File(...)):
    """Handles file uploads and stores them on disk."""
    try:
        filename = f"{session_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(await file.read())

        # Optional: associate file with session
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        conversation_history[session_id].append({
            "role": "system",
            "content": f"[User uploaded file: {file.filename}]",
            "file_path": filepath
        })

        return JSONResponse({"success": True, "filename": file.filename, "path": filepath})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)



@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "anonymous")  # default fallback
    model  = data.get("model") or DEFAULT_MODEL
    stream = bool(data.get("stream", True))

    # 🧩 Ensure the session exists
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    # 🧠 Check for topic change
    if detect_topic_change(session_id, user_message):
        # Reset history but keep summary for context continuity
        conversation_summary[session_id] = conversation_summary.get(session_id, "")
        conversation_history[session_id] = []
        print(f"🔄 Topic shift detected for {session_id} → Resetting message history.")

    # Add user message to session history
    conversation_history[session_id].append({"role": "user", "content": user_message})

    # Possibly summarize if too long
    asyncio.create_task(summarize_conversation(session_id))

    # Build the combined prompt from all messages
    full_prompt = build_prompt(session_id, user_message)

    # --- Stream Mode ---
    if stream:
        async def sse():
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
                        except Exception:
                            yield f"data: {json.dumps({'delta': line})}\n\n"
                            continue

                        # Handle API errors
                        if payload.get("error"):
                            yield f"data: {json.dumps({'error': payload['error']})}\n\n"
                            break

                        # Stream LLM responses as they arrive
                        if "response" in payload:
                            chunk = payload["response"]
                            response_text += chunk
                            yield f"data: {json.dumps({'delta': chunk})}\n\n"

                        if payload.get("done"):
                            break

            # 🧠 Save assistant reply to memory
            conversation_history[session_id].append(
                {"role": "assistant", "content": response_text.strip()}
            )

            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")


    # --- Non-stream Mode ---
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
        )
    r.raise_for_status()
    out = r.json()
    text = out.get("response", "").strip()

    # 🧠 Save assistant reply
    conversation_history[session_id].append({"role": "assistant", "content": text})

    return JSONResponse({"text": text, "sources": []})
