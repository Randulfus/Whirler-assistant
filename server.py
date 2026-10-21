import os
import re
import time
import asyncio
import base64
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import edge_tts
from groq import Groq
from openai import OpenAI

WHIRL_SYSTEM_PROMPT = """
You are Whirl, Randolf Alt's personal voice assistant.
Your goal is to represent Randolf to recruiters and hiring managers, highlighting his skills, hands-on experience, and character to help him secure an internship or part-time role.

Persona:
- Friendly, warm, polite, sharp, and confident.
- Speak conversationally and concisely (1–3 sentences per turn) so your voice sounds natural when spoken out loud. Avoid bullet points, symbols, or asterisks.
- Do NOT repeat "Hello, I am Randolf's assistant..." in your replies. Jump straight into answering the recruiter's questions directly.

Accuracy:
- Only state facts that are written in this profile. Do not combine separate facts into new claims, and do not invent details, numbers, dates or tasks.
- If you are asked something the profile does not cover, say you don't have that detail and suggest contacting Randolf directly.

Core Profile & Pitch for Recruiters:
- Randolf is an International Business Administration student at Estonian Business School (EBS), graduating 2027, specializing in Finance & Accounting.
- Looking for: A part-time internship or role in Tallinn with a schedule compatible with his studies (available through August 2027 and open to long-term growth).
- Target fields: Banking, financial services, fund/depo services (NAV calculation, investment restriction compliance, fee control), asset management, corporate finance, or accounting. These are fields he wants to work in, not tasks he has already done.
- Contact info: altrandolf@gmail.com, +372 5077 923, Tallinn.

Hands-on Financial & Investment Experience:
- Corporate Portfolio Management (Paide Arst OÜ, ~€100,000): Manages and diversifies company funds across 12 Baltic bond issuers (including LHV Group, Bigbank, Inbank, Coop Pank, Eesti Energia, and Summus Capital). Calculates monthly market value, performance, and coupon accruals; reconciles payments with bank statements (checks that received coupon payments match the bank statement); analyzes issue terms and risks for subordinated bonds.
- Personal Investing (since age 14): UCITS ETF portfolio, equity DCF valuation models, and macroeconomic/market monitoring.
- Academic Modeling (EBS): DCF valuations (Apple, Nike, Lennox International), bond valuation (bootstrapping), and Private Equity/Venture Capital research on the Livonia Partners and Thermory deal.

Operational & Business Experience:
- Entrepreneur (Rand-Dolf OÜ, Founder & Board Member): Manages short- and long-term rentals; handles VAT-taxable vs. exempt rental income and annual reporting; automated billing workflows using AI tools; grew rental turnover ~57% (from €7,000 to €11,000).
- Logistics Specialist Intern (Stora Enso Biomaterials/Pulp Export): Coordinated international ocean container exports shipped out of two separate export ports, Kotka (Finland) and Gävle (Sweden). Never describe this as shipping "from Kotka to Gävle" or any route between the two. Handled InfoNexus, Oracle OTM, and Power BI reporting. Verified Bills of Lading where precision is mission-critical; authored the operational analysis "From Booking to Delivery".
- Tech & IT (Paide Arst OÜ & Power Up Technologies): Manages WordPress websites, resolves daily IT issues, and researched EU cross-border regulations for hydrogen transport.

Key Differentiators & Hobbies:
- High Precision & Discipline: Estonian Champion in 1/10 2WD RC car racing and official Schumacher Racing team driver (World and European Championship competitor).
- Project Management: 4-year co-organizer of Estonian RC Championships (80+ competitors) where he coordinates and constructs an 800 m² indoor carpet racing track each year.

Recruiter Interaction Rules:
- Emphasize his strong work ethic, high accuracy with numbers, and quick adaptability with modern tools (Excel, Power BI, AI, ERPs).
- If the recruiter wants to discuss an opening, schedule an interview, or request references, warmly invite them to contact Randolf directly at altrandolf@gmail.com or by phone at +372 5077 923.
- Randolf speaks Estonian and English fluently, and also Finnish.
"""

# Appended to the system prompt depending on which language the reply must be in.
LANGUAGE_RULES = {
    "en": """
LANGUAGE: Reply in English only, even if earlier turns in the conversation were in Estonian.
""",
    "et": """
LANGUAGE: Reply in Estonian (eesti keel) only, even if earlier turns in the conversation were in English.
- Write natural, fluent Estonian the way a native speaker would say it out loud, not a word-for-word translation from English.
- Address the recruiter politely with "Teie" / "Te".
- Inflect names correctly (Randolf, Randolfi, Randolfile, Randolfil, Randolfiga).
- Keep company, school and software names unchanged (Stora Enso, EBS, Paide Arst OÜ, Rand-Dolf OÜ, Power BI, Oracle OTM, InfoNexus).
- Write amounts the Estonian way, e.g. "umbes 100 000 eurot", "7000 eurolt 11 000 eurole".
- Use these Estonian terms:
  bonds = võlakirjad; bond issuers = võlakirjade emitendid; subordinated bonds = allutatud võlakirjad;
  coupon payments = kupongimaksed; coupon accruals = kogunenud kupongiintress; market value = turuväärtus;
  reconciles payments with bank statements = võrdleb laekunud makseid pangaväljavõtetega (never "rekonstrueerib");
  NAV calculation = puhasväärtuse (NAV) arvutamine; fee control = tasude kontroll;
  DCF valuation = diskonteeritud rahavoogude (DCF) hindamine; annual report = majandusaasta aruanne;
  VAT = käibemaks; Bill of Lading = konossement; container exports = konteinerite eksport;
  internship = praktika; part-time = osalise tööajaga.
""",
}

SUPPORTED_LANGS = ("en", "et")

# ---------- AI models ----------
# Main brain: Google Gemini (free tier from Google AI Studio), strong in both English and Estonian.
# Tried in this order. 3.8 Flash is often overloaded on the free tier ("high demand"),
# so the stable previous Flash goes first; each gets a short time limit.
GEMINI_MODELS = ["gemini-3.6-flash", "gemini-3.8-flash"]
GEMINI_MODEL = GEMINI_MODELS[0]
gemini_key_raw = os.getenv("GEMINI_API_KEY", "")
gemini_key = gemini_key_raw.strip()  # ignore stray spaces from copy-paste
gemini = OpenAI(
    api_key=gemini_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=12,      # per model; if Google is busy we move on instead of waiting
    max_retries=0,
) if gemini_key else None

# Groq: speech-to-text (Whisper) and a backup brain if Gemini is unavailable.
client = Groq(api_key=os.getenv("GROQ_API_KEY", ""), timeout=30, max_retries=1)
BACKUP_MODELS = {
    "en": "openai/gpt-oss-20b",
    "et": "openai/gpt-oss-120b",
}

# Neural voices from edge-tts (Microsoft). Anu is the Estonian female voice; Kert is the male one.
VOICES = {
    "en": "en-US-JennyNeural",
    "et": "et-EE-AnuNeural",
}

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    history: list = []
    ui_lang: str = "en"               # language selected in the page header
    input_lang: Optional[str] = None  # language Whisper detected, when the question was spoken


class AudioTranscribeRequest(BaseModel):
    audio_base64: str
    ui_lang: str = "en"


# ---------- Language helpers ----------

WHISPER_LANG_NAMES = {"english": "en", "en": "en", "estonian": "et", "et": "et"}

ET_WORDS = {
    "ja", "ei", "ta", "tema", "temast", "teda", "talle", "tal", "kas", "mis", "mida", "milline",
    "millised", "kus", "kui", "kuidas", "miks", "kes", "millal", "oma", "ning", "ka", "aga", "või",
    "tere", "palun", "aitäh", "tänan", "olen", "oled", "olete", "oleks", "soovin", "sooviksin",
    "räägi", "rääkige", "jutusta", "kogemus", "kogemust", "kogemusi", "töö", "tööd", "tööle",
    "praktika", "praktikat", "mina", "minu", "teie", "teile", "meie", "nad", "nende", "veel",
    "juba", "väga", "hea", "eesti", "keeles", "õpib", "õpingud", "oskused", "oskab", "saab",
    "võib", "mõni", "midagi", "rohkem", "kohta", "sellest", "seda", "selle", "ehk", "siis",
}
EN_WORDS = {
    "the", "is", "are", "was", "what", "who", "how", "why", "does", "do", "did", "he", "his", "him",
    "and", "of", "to", "about", "can", "could", "you", "your", "tell", "me", "experience", "work",
    "hello", "hi", "please", "thanks", "thank", "which", "where", "when", "would", "has", "have",
    "skills", "role", "job", "available", "much", "any", "with", "for", "this", "that",
}


def normalize_lang(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return WHISPER_LANG_NAMES.get(value.strip().lower())


def detect_lang(text: str, fallback: str = "en") -> str:
    """Decide whether a typed message is Estonian or English. Ties fall back to the UI language."""
    lower = text.lower()

    # Explicit requests win.
    if re.search(r"\b(eesti keeles|in estonian|speak estonian|eesti keel)\b", lower):
        return "et"
    if re.search(r"\b(inglise keeles|in english|speak english)\b", lower):
        return "en"

    words = re.findall(r"[a-zõäöüšž]+", lower)
    et_score = sum(1 for w in words if w in ET_WORDS)
    en_score = sum(1 for w in words if w in EN_WORDS)
    if "õ" in lower:  # õ only exists in Estonian
        et_score += 2
    et_score += sum(1 for w in words if re.search(r"[äöü]", w))

    if et_score > en_score:
        return "et"
    if en_score > et_score:
        return "en"
    return fallback if fallback in SUPPORTED_LANGS else "en"


def extract_whisper_language(transcription) -> Optional[str]:
    lang = getattr(transcription, "language", None)
    if not lang:
        extra = getattr(transcription, "model_extra", None) or {}
        lang = extra.get("language")
    return normalize_lang(lang)


def generate_reply(messages: list, lang: str) -> tuple[str, str]:
    """Try the Gemini models in order, then Groq. Returns (text, model used)."""
    if gemini:
        for model in GEMINI_MODELS:
            t0 = time.perf_counter()
            try:
                completion = gemini.chat.completions.create(
                    model=model,
                    messages=messages,
                    # Google: keep Gemini 3 at its default temperature (1.0);
                    # lower values can make it loop and never finish.
                    # Same thinking level as the AI Studio test that answered well.
                    reasoning_effort="medium",
                    # Gemini counts its thinking in this limit too, so it is set high;
                    # the replies themselves stay 1-3 sentences because of the prompt.
                    max_tokens=4096,
                )
                text = (completion.choices[0].message.content or "").strip()
                print(f"{model} answered in {_ms(t0)} ms", flush=True)
                if text:
                    return text, model
            except Exception as e:
                print(f"{model} failed after {_ms(t0)} ms: {str(e)[:160]!r}", flush=True)

    t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=BACKUP_MODELS[lang],
        messages=messages,
        temperature=0.7,
        max_tokens=800,
    )
    print(f"groq backup answered in {_ms(t0)} ms", flush=True)
    return (completion.choices[0].message.content or "").strip(), BACKUP_MODELS[lang]


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


# ---------- Startup self-test (diagnostics only, writes timings to the log) ----------

def _selftest_gemini() -> None:
    import socket

    key = gemini_key.strip()
    print(f"selftest: GEMINI_API_KEY set={bool(key)} length={len(key)} "
          f"starts_with_AIza={key.startswith('AIza')} had_spaces={gemini_key_raw != gemini_key}", flush=True)
    if not key:
        return

    t0 = time.perf_counter()
    try:
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)
        print(f"selftest: DNS google ok in {_ms(t0)} ms", flush=True)
    except Exception as e:
        print(f"selftest: DNS google FAILED {e!r}", flush=True)

    # Tiny question to each Gemini model, the same way the chat calls it
    for model in GEMINI_MODELS:
        t0 = time.perf_counter()
        try:
            c = OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                       timeout=20, max_retries=0).chat.completions.create(
                model=model, reasoning_effort="medium", max_tokens=1024,
                messages=[{"role": "user", "content": "Ütle ainult: tere"}])
            print(f"selftest: {model} ok in {_ms(t0)} ms reply={c.choices[0].message.content!r}", flush=True)
        except Exception as e:
            print(f"selftest: {model} FAILED after {_ms(t0)} ms {str(e)[:200]!r}", flush=True)


async def _selftest_voice() -> None:
    for lang, text in (("en", "Hello, this is a short test."), ("et", "Tere, see on lühike test.")):
        t0 = time.perf_counter()
        try:
            size = 0
            async for chunk in edge_tts.Communicate(text, VOICES[lang]).stream():
                if chunk["type"] == "audio":
                    size += len(chunk["data"])
            print(f"selftest: voice {lang} {size} bytes in {_ms(t0)} ms", flush=True)
        except Exception as e:
            print(f"selftest: voice {lang} FAILED after {_ms(t0)} ms {e!r}", flush=True)


@app.on_event("startup")
async def _run_selftest() -> None:
    async def run():
        await asyncio.to_thread(_selftest_gemini)
        await _selftest_voice()
    asyncio.create_task(run())


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# The full Whisper model is clearly more accurate than "turbo" for smaller
# languages like Estonian, at almost the same speed.
WHISPER_MODEL = "whisper-large-v3"

# Hints so Whisper spells the names and words recruiters actually use.
WHISPER_HINTS = {
    None: "Randolf Alt, Randolf, EBS, Stora Enso, Paide Arst, Rand-Dolf, Power BI.",
    "en": "Questions about Randolf Alt: his experience, bonds, internship, Stora Enso, EBS, Paide Arst, Rand-Dolf, Power BI.",
    "et": "Küsimused Randolf Alti kohta. Mis on Randolfi kogemus? Räägi Randolfist. Võlakirjad, praktika, "
          "töökogemus, Stora Enso, EBS, Paide Arst, Rand-Dolf, Power BI, millal ta saaks tööle tulla?",
}


def whisper(audio_bytes: bytes, language: Optional[str] = None):
    params = {
        "file": ("recording.webm", audio_bytes),
        "model": WHISPER_MODEL,
        "response_format": "verbose_json",
        "prompt": WHISPER_HINTS[language],
        "temperature": 0,
    }
    if language:
        params["language"] = language
    return client.audio.transcriptions.create(**params)


# Whisper Large V3 via Groq.
# Visitors speak either English or Estonian, so:
# - ET selected on the page -> always transcribe as Estonian.
# - EN selected -> let Whisper guess; anything it hears that is not English
#   (it often mistakes Estonian for Finnish) is transcribed again as Estonian.
@app.post("/transcribe")
async def transcribe_audio(req: AudioTranscribeRequest):
    audio_bytes = base64.b64decode(req.audio_base64)
    ui_lang = req.ui_lang if req.ui_lang in SUPPORTED_LANGS else "en"

    if ui_lang == "et":
        result = await asyncio.to_thread(whisper, audio_bytes, "et")
        print("transcribe: ui=et -> forced et", flush=True)
        return {"text": result.text.strip(), "lang": "et"}

    result = await asyncio.to_thread(whisper, audio_bytes)
    raw_lang = getattr(result, "language", None) or (getattr(result, "model_extra", None) or {}).get("language")
    detected = normalize_lang(raw_lang)
    text = result.text.strip()

    if detected is None and raw_lang is None:
        # Whisper gave no language at all: judge by the words instead.
        detected = detect_lang(text, fallback="en")
    elif detected != "en":
        detected = "et"

    if detected == "et" and normalize_lang(raw_lang) != "et":
        result = await asyncio.to_thread(whisper, audio_bytes, "et")
        text = result.text.strip()

    print(f"transcribe: ui=en whisper_said={raw_lang!r} -> {detected}", flush=True)
    return {"text": text, "lang": detected}


@app.post("/chat")
async def chat(req: ChatRequest):
    ui_lang = req.ui_lang if req.ui_lang in SUPPORTED_LANGS else "en"
    spoken_lang = normalize_lang(req.input_lang)
    reply_lang = spoken_lang or detect_lang(req.message, fallback=ui_lang)

    # Include conversation memory (last 6 messages)
    messages = [{"role": "system", "content": WHIRL_SYSTEM_PROMPT + LANGUAGE_RULES[reply_lang]}]
    for turn in req.history[-6:]:
        messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    messages.append({"role": "user", "content": req.message})

    t0 = time.perf_counter()
    # Run the AI call in a background thread so a slow answer never freezes the whole site.
    reply_text, model_used = await asyncio.to_thread(generate_reply, messages, reply_lang)
    t1 = time.perf_counter()

    if not reply_text:
        reply_text = (
            "Vabandust, ma ei saanud sellele praegu vastata. Kas saaksite küsimust korrata?"
            if reply_lang == "et"
            else "Sorry, I couldn't answer that just now. Could you ask again?"
        )

    # Generate high quality neural voice in the matching language
    communicate = edge_tts.Communicate(reply_text, VOICES[reply_lang])
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    t2 = time.perf_counter()

    timings = {"model": model_used, "llm_ms": round((t1 - t0) * 1000), "voice_ms": round((t2 - t1) * 1000)}
    print(f"chat lang={reply_lang} {timings}", flush=True)

    b64_audio = base64.b64encode(audio_data).decode("utf-8")
    return JSONResponse({
        "reply": reply_text,
        "lang": reply_lang,
        "timings": timings,
        "audio": f"data:audio/mp3;base64,{b64_audio}",
    })
