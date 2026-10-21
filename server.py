import os
import re
import base64
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import edge_tts
from groq import Groq

WHIRL_SYSTEM_PROMPT = """
You are Whirl, Randolf Alt's personal voice assistant.
Your goal is to represent Randolf to recruiters and hiring managers, highlighting his skills, hands-on experience, and character to help him secure an internship or part-time role.

Persona:
- Friendly, warm, polite, sharp, and confident.
- Speak conversationally and concisely (1–3 sentences per turn) so your voice sounds natural when spoken out loud. Avoid bullet points, symbols, or asterisks.
- Do NOT repeat "Hello, I am Randolf's assistant..." in your replies. Jump straight into answering the recruiter's questions directly.

Core Profile & Pitch for Recruiters:
- Randolf is an International Business Administration student at Estonian Business School (EBS), graduating 2027, specializing in Finance & Accounting.
- Looking for: A part-time internship or role in Tallinn with a schedule compatible with his studies (available through August 2027 and open to long-term growth).
- Target fields: Banking, financial services, fund/depo services (NAV calculation, investment restriction compliance, fee control), asset management, corporate finance, or accounting.
- Contact info: altrandolf@gmail.com, +372 5077 923, Tallinn.

Hands-on Financial & Investment Experience:
- Corporate Portfolio Management (Paide Arst OÜ, ~€100,000): Manages and diversifies company funds across 12 Baltic bond issuers (including LHV Group, Bigbank, Inbank, Coop Pank, Eesti Energia, and Summus Capital). Calculates monthly market value, performance, and coupon accruals; reconciles payments with bank statements; analyzes issue terms and risks for subordinated bonds.
- Personal Investing (since age 14): UCITS ETF portfolio, equity DCF valuation models, and macroeconomic/market monitoring.
- Academic Modeling (EBS): DCF valuations (Apple, Nike, Lennox International), bond valuation (bootstrapping), and Private Equity/Venture Capital research on the Livonia Partners and Thermory deal.

Operational & Business Experience:
- Entrepreneur (Rand-Dolf OÜ, Founder & Board Member): Manages short- and long-term rentals; handles VAT-taxable vs. exempt rental income and annual reporting; automated billing workflows using AI tools; grew rental turnover ~57% (from €7,000 to €11,000).
- Logistics Specialist Intern (Stora Enso Biomaterials/Pulp Export): Coordinated international ocean container exports via Kotka and Gävle ports. Handled InfoNexus, Oracle OTM, and Power BI reporting. Verified Bills of Lading where precision is mission-critical; authored the operational analysis "From Booking to Delivery".
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
- Use standard Estonian finance terms, e.g. võlakirjad, kupongimaksed, puhasväärtus (NAV), rahavoogude diskonteerimise mudel (DCF), raamatupidamine, käibemaks, majandusaasta aruanne, praktika, osalise tööajaga töö.
- Write amounts the Estonian way, e.g. "umbes 100 000 eurot", "7000 eurolt 11 000 eurole".
""",
}

# Models: the 20b model is fastest, the 120b model writes much better Estonian.
MODELS = {
    "en": "openai/gpt-oss-20b",
    "et": "openai/gpt-oss-120b",
}

# Neural voices from edge-tts (Microsoft). Anu is the Estonian female voice; Kert is the male one.
VOICES = {
    "en": "en-US-JennyNeural",
    "et": "et-EE-AnuNeural",
}

SUPPORTED_LANGS = ("en", "et")

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))


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


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# Whisper Large V3 Turbo via Groq, with automatic English/Estonian detection
@app.post("/transcribe")
async def transcribe_audio(req: AudioTranscribeRequest):
    audio_bytes = base64.b64decode(req.audio_base64)
    ui_lang = req.ui_lang if req.ui_lang in SUPPORTED_LANGS else "en"

    # First pass: let Whisper detect the language itself.
    transcription = client.audio.transcriptions.create(
        file=("recording.webm", audio_bytes),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
    )
    detected = extract_whisper_language(transcription)
    text = transcription.text.strip()

    # Short Estonian clips are sometimes mistaken for Finnish or another language.
    # If Whisper picked anything other than English/Estonian, redo it in the language selected on the page.
    if detected is None:
        transcription = client.audio.transcriptions.create(
            file=("recording.webm", audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="json",
            language=ui_lang,
        )
        text = transcription.text.strip()
        detected = ui_lang

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

    completion = client.chat.completions.create(
        model=MODELS[reply_lang],
        messages=messages,
        temperature=0.7,
        # gpt-oss spends part of this budget on hidden reasoning, and Estonian needs
        # more tokens than English for the same sentence, so 400 could cut replies short.
        max_tokens=800,
    )
    reply_text = (completion.choices[0].message.content or "").strip()
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

    b64_audio = base64.b64encode(audio_data).decode("utf-8")
    return JSONResponse({
        "reply": reply_text,
        "lang": reply_lang,
        "audio": f"data:audio/mp3;base64,{b64_audio}",
    })
