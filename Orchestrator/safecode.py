from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx, os, json, asyncio

app = FastAPI()

# Internal service URLs / defaults
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DEFAULT_MODEL   = os.getenv("DEFAULT_MODEL", "llama2:7b")

# 🧠 In-memory storage for conversations { session_id: [ {"role": "user"/"assistant", "content": "..."} ] }
conversation_history = {}

def build_prompt(session_id: str, user_message: str) -> str:
    """Builds a structured prompt using past conversation without repetition"""
    history = conversation_history.get(session_id, [])

    # 🧩 Clear system instruction to avoid repeating
    formatted = (
        "You are a helpful AI assistant. Use the conversation history below to stay consistent, "
        "but do NOT repeat or restate previous messages. Respond naturally and concisely.\n\n"
    )

    # Include the last few exchanges for context
    for msg in history[-10:]:
        role = msg["role"].capitalize()
        content = msg["content"]
        formatted += f"{role}: {content}\n"

    # Add the user's new input at the end
    formatted += f"User: {user_message}\nAssistant:"
    return formatted


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

    # Add user message to session history
    conversation_history[session_id].append({"role": "user", "content": user_message})

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
