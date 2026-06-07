from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import Response
import requests
import os
import time
from dotenv import load_dotenv
from glossary import GlossaryMatcher

load_dotenv()
 
app = FastAPI(title="Indic Voice Translator API")
 
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
 
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
glossary = GlossaryMatcher()
 
HF_SEAMLESS = "https://api-inference.huggingface.co/models/facebook/seamless-m4t-v2-large"
 
@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}
 
@app.post("/translate-voice")
async def translate_voice(
    file: UploadFile,
    src: str = Form(),
    tgt: str = Form()
):
    allowed_langs = ["hi", "en", "kn", "te"]
    if src not in allowed_langs or tgt not in allowed_langs:
        raise HTTPException(400, f"Unsupported lang. Use one of {allowed_langs}")
 
    if src == tgt:
        raise HTTPException(400, "Source and target language cannot be same")
 
    audio_bytes = await file.read()
 
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")
 
    if len(audio_bytes) > 2 * 1024 * 1024:
        raise HTTPException(400, "Audio too large. Keep recordings under 10 seconds")
 
    # Call HF model with retry
    params = {"src_lang": src, "tgt_lang": tgt}
 
    for attempt in range(2):
        try:
            resp = requests.post(
                HF_SEAMLESS,
                headers=HEADERS,
                data=audio_bytes,
                params=params,
                timeout=30
            )
 
            if resp.status_code == 200:
                return Response(content=resp.content, media_type="audio/wav")
 
            if resp.status_code == 503:
                time.sleep(2)
                continue
 
            raise HTTPException(resp.status_code, f"HF error: {resp.text}")
 
        except requests.Timeout:
            if attempt == 1:
                raise HTTPException(504, "Request timed out")
            time.sleep(2)
 
    raise HTTPException(500, "Failed after 2 attempts")