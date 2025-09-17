from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx, os, json

app = FastAPI()

# Internal service URLs / defaults
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DEFAULT_MODEL   = os.getenv("DEFAULT_MODEL",   "llama3.2:3b")  # <= default to 3B

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    model  = data.get("model") or DEFAULT_MODEL
    stream = bool(data.get("stream", True))

    if stream:
        async def sse():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": model, "prompt": user_message, "stream": True},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            # Forward any non-JSON as raw delta
                            yield f"data: {json.dumps({'delta': line})}\n\n"
                            continue

                        # Surface upstream errors to the client
                        if payload.get("error"):
                            yield f"data: {json.dumps({'error': payload['error']})}\n\n"
                            break

                        # Normal token delta
                        if "response" in payload:
                            yield f"data: {json.dumps({'delta': payload['response']})}\n\n"

                        if payload.get("done"):
                            break
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # Non-streaming (single JSON doc; jq-friendly)
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": user_message, "stream": False},
        )
    r.raise_for_status()
    out = r.json()
    return JSONResponse({"text": out.get("response", ""), "sources": []})
