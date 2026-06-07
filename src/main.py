from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
import requests
import os
import time
import logging

from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Add it in Render → Environment.")

HF_SEAMLESS = "https://api-inference.huggingface.co/models/facebook/seamless-m4t-v2-large"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

ALLOWED_LANGS = {"hi", "en", "kn", "te", "mr", "bn", "ta", "gu", "ml"}

MAX_AUDIO_BYTES = 2 * 1024 * 1024   # 2 MB  (~10-15 s of speech)
HF_TIMEOUT      = 45                 # seconds — HF cold-start can be slow
HF_RETRIES      = 3

# ── Lazy-loaded glossary ────────────────────────────────────────────────────────
# NOT loaded at import time — that would consume ~1.5 GB RAM on boot and
# kill the process on Render's 512 MB free tier.
_glossary = None

def get_glossary():
    """Load GlossaryMatcher on first use, then cache it."""
    global _glossary
    if _glossary is None:
        log.info("Loading GlossaryMatcher (first request)…")
        try:
            from glossary import GlossaryMatcher
            _glossary = GlossaryMatcher()
            log.info("GlossaryMatcher ready.")
        except Exception as exc:
            log.warning("GlossaryMatcher failed to load: %s — continuing without it.", exc)
            _glossary = _NoOpGlossary()
    return _glossary


class _NoOpGlossary:
    """Fallback when the glossary can't be loaded (missing file, OOM, etc.)."""
    def replace(self, text, src, tgt):
        return text


# ── Lifespan (replaces deprecated @app.on_event) ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Indic Voice Translator API starting up.")
    # Intentionally NOT pre-loading the glossary here — keep boot fast & light.
    yield
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Indic Voice Translator API",
    description="Speech-to-speech translation between Indian languages via SeamlessM4T v2.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    """Lightweight health check — safe to poll every few minutes to prevent cold starts."""
    return {"status": "ok", "time": time.time()}


@app.get("/languages", tags=["meta"])
def languages():
    """Return the list of supported BCP-47 language codes."""
    return {
        "supported": sorted(ALLOWED_LANGS),
        "details": {
            "hi": "Hindi",
            "en": "English",
            "kn": "Kannada",
            "te": "Telugu",
            "mr": "Marathi",
            "bn": "Bengali",
            "ta": "Tamil",
            "gu": "Gujarati",
            "ml": "Malayalam",
        },
    }


@app.post("/translate-voice", tags=["translation"])
async def translate_voice(
    file: UploadFile,
    src: str = Form(..., description="Source language code, e.g. 'hi'"),
    tgt: str = Form(..., description="Target language code, e.g. 'kn'"),
):
    """
    Translate a voice clip from one Indian language to another.

    - **file**: WAV/MP3 audio (mono, 16 kHz recommended). Keep under ~15 seconds.
    - **src**: BCP-47 language code of the input audio.
    - **tgt**: BCP-47 language code for the translated output.

    Returns raw WAV bytes on success.
    """

    # ── Input validation ───────────────────────────────────────────────────────
    src = src.strip().lower()
    tgt = tgt.strip().lower()

    if src not in ALLOWED_LANGS or tgt not in ALLOWED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Allowed codes: {sorted(ALLOWED_LANGS)}",
        )

    if src == tgt:
        raise HTTPException(
            status_code=400,
            detail="Source and target language must be different.",
        )

    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Audio too large ({len(audio_bytes) // 1024} KB). "
                   f"Maximum allowed is {MAX_AUDIO_BYTES // 1024} KB (~10–15 seconds).",
        )

    log.info("Translate request: %s → %s, file=%s, size=%d bytes",
             src, tgt, file.filename, len(audio_bytes))

    # ── HuggingFace inference call (with retry) ────────────────────────────────
    params  = {"src_lang": src, "tgt_lang": tgt}
    last_err: Exception | None = None

    for attempt in range(1, HF_RETRIES + 1):
        log.info("HF attempt %d/%d", attempt, HF_RETRIES)
        try:
            resp = requests.post(
                HF_SEAMLESS,
                headers=HEADERS,
                data=audio_bytes,
                params=params,
                timeout=HF_TIMEOUT,
            )

            if resp.status_code == 200:
                log.info("HF success on attempt %d, response size=%d bytes",
                         attempt, len(resp.content))
                return Response(content=resp.content, media_type="audio/wav")

            if resp.status_code == 503:
                # Model is loading on HF side — back off and retry
                wait = 5 * attempt
                log.warning("HF 503 (model loading). Waiting %ds before retry…", wait)
                time.sleep(wait)
                last_err = HTTPException(503, "HuggingFace model is warming up. Please retry.")
                continue

            if resp.status_code == 401:
                log.error("HF 401 — HF_TOKEN is invalid or expired.")
                raise HTTPException(
                    status_code=500,
                    detail="Server authentication with HuggingFace failed. Check HF_TOKEN.",
                )

            # Any other non-200 — log and surface it
            log.error("HF error %d: %s", resp.status_code, resp.text[:300])
            raise HTTPException(
                status_code=502,
                detail=f"HuggingFace returned {resp.status_code}: {resp.text[:200]}",
            )

        except requests.Timeout:
            wait = 3 * attempt
            log.warning("HF request timed out (attempt %d). Waiting %ds…", attempt, wait)
            last_err = HTTPException(504, "HuggingFace request timed out.")
            if attempt < HF_RETRIES:
                time.sleep(wait)

        except HTTPException:
            raise  # re-raise our own HTTP errors without wrapping

        except Exception as exc:
            log.exception("Unexpected error calling HuggingFace: %s", exc)
            last_err = HTTPException(500, f"Unexpected error: {exc}")
            if attempt < HF_RETRIES:
                time.sleep(3)

    # All retries exhausted
    if last_err:
        raise last_err
    raise HTTPException(status_code=500, detail="Translation failed after all retries.")