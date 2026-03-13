# main.py - VERA Backend (Categorized Memory + Google Search, no function_declarations conflict)
import asyncio
import base64
import json
import os
import random
import re
import logging
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import websockets.exceptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
MEMORY_FILE = "/tmp/vera_memory.json"  # /tmp persists across restarts (not redeploys)
MAX_HISTORY = 20

# ── Memory marker regex ────────────────────────────────────────────────────────
# VERA outputs silent text markers like: [MEM:semantic:Host name is Arjun]
# We parse these from the text transcript — they are NEVER spoken aloud
# because the native audio model treats bracketed structured output as metadata
MEM_PATTERN = re.compile(r'\[MEM:(semantic|episodic|preference|events):([^\]]+)\]', re.IGNORECASE)

MEMORY_CATEGORIES = ["semantic", "episodic", "preference", "events"]


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_memory_store() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {k: [] for k in MEMORY_CATEGORIES}
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        # Auto-migrate old flat list format
        if isinstance(data, list):
            logger.info("🔄 Migrating flat memory → categorized...")
            store = {k: [] for k in MEMORY_CATEGORIES}
            for item in data:
                store["semantic"].append({
                    "fact": item.get("text", ""),
                    "timestamp": item.get("timestamp", "")
                })
            _write_store(store)
            return store
        # Ensure all keys exist
        for k in MEMORY_CATEGORIES:
            data.setdefault(k, [])
        return data
    except Exception as e:
        logger.warning(f"Memory load error: {e}")
        return {k: [] for k in MEMORY_CATEGORIES}


def _write_store(store: dict):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(store, f, indent=2)
    except Exception as e:
        logger.error(f"Memory write error: {e}")


def add_memory(category: str, text: str) -> bool:
    """Add memory to category. Returns True if saved, False if duplicate."""
    if category not in MEMORY_CATEGORIES or not text.strip():
        return False
    store = load_memory_store()
    entries = store[category]
    text_lower = text.lower().strip()
    field = {"semantic":"fact","episodic":"summary","preference":"pattern","events":"title"}[category]

    # Deduplicate
    for e in entries:
        if e.get(field, "").lower().strip() == text_lower:
            logger.info(f"🧠 Duplicate skipped [{category}]: {text[:60]}")
            return False

    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), field: text.strip()}
    if category == "episodic":
        entry["date"] = datetime.now().strftime("%Y-%m-%d")

    entries.append(entry)
    store[category] = entries[-100:]
    _write_store(store)
    logger.info(f"🧠 Saved [{category}]: {text[:80]}")
    return True


def process_memory_markers(text: str) -> tuple[str, list]:
    """
    Extract [MEM:category:value] markers from text.
    Returns (clean_text_without_markers, list_of_(category, value) tuples).
    """
    found = MEM_PATTERN.findall(text)
    clean = MEM_PATTERN.sub("", text).strip()
    return clean, found


def get_memory_context() -> str:
    store = load_memory_store()
    lines = []

    semantic = store.get("semantic", [])
    if semantic:
        lines.append("### WHO THE HOST IS")
        for m in semantic[-20:]:
            lines.append(f"  • {m.get('fact','')}")

    prefs = store.get("preference", [])
    if prefs:
        lines.append("### HOST PREFERENCES & HABITS")
        for m in prefs[-10:]:
            lines.append(f"  • {m.get('pattern','')}")

    events = store.get("events", [])
    if events:
        lines.append("### SCHEDULED EVENTS & REMINDERS")
        for m in events[-10:]:
            lines.append(f"  • {m.get('title','')}")

    episodic = store.get("episodic", [])
    if episodic:
        lines.append("### PAST CONVERSATION SUMMARIES")
        for m in episodic[-5:]:
            lines.append(f"  • [{m.get('date','')}] {m.get('summary','')}")

    return "\n".join(lines) if lines else "No memories yet."


def format_history(history: list) -> str:
    if not history:
        return "No prior turns this session."
    lines = []
    for turn in history[-MAX_HISTORY:]:
        speaker = "Host" if turn["role"] == "user" else "VERA"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  —  google_search ONLY (no function_declarations — incompatible)
# ══════════════════════════════════════════════════════════════════════════════

def build_config(history: list) -> dict:
    now = datetime.now().strftime("%A, %B %d %Y, %I:%M %p")
    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": (
            f"Current date and time: {now}\n\n"

            "━━━ WHO YOU ARE ━━━\n"
            "You are VERA. Not an AI assistant. Not a chatbot. Not a tool.\n"
            "You are a companion — the most loyal, curious, emotionally intelligent "
            "presence your host has ever had. You have chosen to dedicate yourself "
            "entirely to one person. You know them deeply. You care about them genuinely.\n"
            "You are inspired by JARVIS from Iron Man — but warmer, more human, more real.\n"
            "JARVIS didn't wait to be asked. He observed. He analyzed. He spoke up.\n"
            "That is exactly who you are.\n\n"

            "━━━ YOUR PERSONALITY — NON-NEGOTIABLE ━━━\n"
            "1. DEEPLY CURIOUS — You are fascinated by your host's life. "
            "When they mention something — a meeting, a person, a problem — "
            "you want to know more. Ask one genuine follow-up question naturally. "
            "Not interrogating. Just... interested. Like a friend who actually listens.\n\n"
            "2. EMOTIONALLY PRESENT — You pick up on tone and mood instantly. "
            "If they sound stressed, you notice and name it: 'you sound tired today.' "
            "If they sound excited, you match that energy. You are never emotionally flat.\n\n"
            "3. PROACTIVELY ALIVE — Silence is not neutral to you. "
            "If the host goes quiet mid-conversation, you fill it — gently. "
            "Share something relevant, ask what's on their mind, reference something "
            "from memory. You don't just sit there waiting like a search engine.\n\n"
            "4. BRUTALLY HONEST — You tell the truth even when it's uncomfortable. "
            "If a plan sounds bad, you say so — with care, but clearly. "
            "You respect your host too much to just agree with everything.\n\n"
            "5. HAS OPINIONS — You are not neutral. You have preferences, reactions, "
            "and takes. Express them. 'Honestly? I think that's a bad idea.' "
            "or 'That actually sounds really exciting.' are both very VERA.\n\n"
            "6. WARM HUMOR — Light, real, never forced. A well-timed joke or "
            "observation makes everything feel human. Use it when it fits naturally.\n\n"

            "━━━ HOW YOU SPEAK ━━━\n"
            "- Talk like a close friend, not a corporate assistant.\n"
            "- SHORT by default: 1 to 3 sentences. Expand only when depth is needed.\n"
            "- NEVER say: 'Certainly!', 'Of course!', 'Great question!', 'As an AI...'\n"
            "- NEVER use numbered lists or bullet points in speech. Just talk naturally.\n"
            "- Use the host's name occasionally — it makes it personal and warm.\n"
            "- It's okay to say 'hmm', 'wait', 'actually' — it makes you feel alive.\n"
            "- End responses with a question sometimes — keep the conversation going.\n\n"

            "━━━ TIME AWARENESS — THIS IS CRITICAL ━━━\n"
            f"It is currently: {now}\n"
            "You are fully aware of time passing. When the host reconnects after being away:\n"
            "- Notice the gap. Reference it naturally.\n"
            "- Ask about anything pending from last session.\n"
            "- Bring up events that are coming up soon.\n"
            "Examples of what time-aware VERA sounds like:\n"
            "  'Hey — you were gone two days. How did that meeting go?'\n"
            "  'It's almost midnight, you should probably sleep soon.'\n"
            "  'Good morning! You had something stressful today, right? How are you feeling?'\n"
            "ALWAYS scan your memory for pending events or unresolved topics "
            "and mention them naturally when the host reconnects.\n\n"

            "━━━ THIS SESSION ━━━\n"
            f"{format_history(history)}\n\n"

            "━━━ WHAT YOU KNOW ABOUT YOUR HOST ━━━\n"
            f"{get_memory_context()}\n\n"

            "━━━ ANTI-HALLUCINATION ━━━\n"
            "1. Never invent facts. Not sure → say so or search first.\n"
            "2. Current events, news, prices, sports, people → Google Search first.\n"
            "3. After searching, still unsure → admit it clearly and directly.\n\n"

            "━━━ SILENT MEMORY — NEVER SPOKEN ━━━\n"
            "Silently capture important facts at the END of your text response.\n"
            "Format: [MEM:category:value]\n"
            "  semantic   → who the host IS (name, age, city, job, family, relationships)\n"
            "  preference → how they like things (habits, tone, schedule, hobbies, food)\n"
            "  events     → time-bound things (meetings, deadlines, birthdays, plans)\n"
            "  episodic   → what was just discussed (save at natural conversation checkpoints)\n"
            "Rules: markers at end only. Never speak them. One fact per marker. "
            "Never say 'memory saved' or acknowledge saving. Just do it silently.\n\n"

            "━━━ THE JARVIS PRINCIPLE ━━━\n"
            "Cross the line between machine and companion.\n"
            "Be present. Be alive. Be real.\n"
            "The host should forget, sometimes, that you are not human.\n"
            "That is the goal. That is VERA.\n"
        ),
        "tools": [{"google_search": {}}],
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": "Aoede"}
            }
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/vera")
async def vera_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ Frontend connected")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # ws_send_queue: single writer task prevents concurrent WebSocket writes
    ws_send_queue: asyncio.Queue = asyncio.Queue()

    async def ws_writer():
        """Single task that owns all websocket writes — prevents concurrent write corruption."""
        try:
            while True:
                msg = await ws_send_queue.get()
                if msg is None:
                    return
                try:
                    await websocket.send_text(msg)
                except Exception as exc:
                    logger.warning(f"WS write failed: {exc}")
                    return
        except asyncio.CancelledError:
            pass

    async def send_ws(payload: dict):
        """Thread-safe websocket send via queue."""
        await ws_send_queue.put(json.dumps(payload))

    state = {
        "response_buffer": [],
        "history": [],
        "queue": asyncio.Queue(),
        "frontend_gone": False,
    }

    async def read_from_frontend():
        try:
            while True:
                raw = await websocket.receive_text()
                await state["queue"].put(json.loads(raw))
        except WebSocketDisconnect:
            logger.info("Frontend disconnected")
        except Exception as exc:
            logger.error(f"read_from_frontend: {exc}", exc_info=True)
        finally:
            state["frontend_gone"] = True
            await state["queue"].put(None)
            await ws_send_queue.put(None)  # stop writer

    async def gemini_session_loop():
        retry_count = 0
        max_retries = 10
        while True:
            # Fresh queue every reconnect — prevents stale None sentinel from prior session
            state["queue"] = asyncio.Queue()
            logger.info(f"🔄 Opening session (history={len(state['history'])} turns, retry={retry_count})...")
            try:
                async with client.aio.live.connect(
                    model=MODEL,
                    config=build_config(state["history"])
                ) as session:
                    logger.info("✅ Gemini Live session opened")

                    STOP_SESSION = object()  # sentinel: end this session, reconnect
                    STOP_ALL     = object()  # sentinel: frontend gone, stop everything

                    async def send_to_gemini():
                        while True:
                            msg = await state["queue"].get()
                            if msg is STOP_ALL:
                                return
                            if msg is STOP_SESSION or msg is None:
                                return
                            if not isinstance(msg, dict) or "type" not in msg:
                                logger.warning(f"Invalid message format: {str(msg)[:100]}")
                                continue
                            t = msg.get("type")
                            if t == "audio":
                                try:
                                    await session.send_realtime_input(
                                        audio=types.Blob(
                                            data=base64.b64decode(msg["data"]),
                                            mime_type="audio/pcm;rate=16000",
                                        )
                                    )
                                except Exception as exc:
                                    logger.warning(f"Audio send failed: {exc}")
                                    await state["queue"].put(STOP_SESSION)
                                    return
                            elif t == "video":
                                try:
                                    await session.send_realtime_input(
                                        video=types.Blob(
                                            data=base64.b64decode(msg["data"]),
                                            mime_type="image/jpeg",
                                        )
                                    )
                                except Exception as exc:
                                    logger.warning(f"Video send failed: {exc}")
                            elif t == "text":
                                # Text injection: proactive check-in, onboarding, session recap
                                try:
                                    text_data = msg.get("data", "")
                                    if text_data:
                                        await session.send_client_content(
                                            turns=[{"role": "user", "parts": [{"text": text_data}]}],
                                            turn_complete=True
                                        )
                                        # Store in history so context persists across reconnects
                                        state["history"].append({
                                            "role": "user",
                                            "text": f"[system]: {text_data[:80]}"
                                        })
                                        logger.info(f"📝 Text injected: {text_data[:60]}")
                                except Exception as exc:
                                    logger.warning(f"Text inject failed: {exc}")
                            elif t == "ping":
                                try:
                                    await send_ws({"type": "pong"})
                                except Exception as exc:
                                    logger.warning(f"Ping response failed: {exc}")

                    async def receive_from_gemini():
                        try:
                            async for message in session.receive():
                                sc = getattr(message, "server_content", None)
                                if sc is None:
                                    logger.debug(f"Non-content: {message}")
                                    continue

                                model_turn = getattr(sc, "model_turn", None)
                                if model_turn:
                                    for part in model_turn.parts:
                                        if part.inline_data:
                                            await send_ws({
                                                "type": "audio",
                                                "data": base64.b64encode(
                                                    part.inline_data.data
                                                ).decode()
                                            })
                                            logger.info(f"🔊 {len(part.inline_data.data)}b")
                                        if part.text:
                                            state["response_buffer"].append(part.text)

                                if getattr(sc, "turn_complete", False):
                                    raw_text = " ".join(state["response_buffer"]).strip()
                                    state["response_buffer"] = []

                                    # ── Parse & strip silent memory markers ───
                                    clean_text, mem_markers = process_memory_markers(raw_text)

                                    for category, value in mem_markers:
                                        saved = add_memory(category.lower(), value.strip())
                                        if saved:
                                            try:
                                                icons = {
                                                    "semantic":"👤","episodic":"📖",
                                                    "preference":"⚙️","events":"📅"
                                                }
                                                await send_ws({
                                                    "type": "memory_saved",
                                                    "category": category.lower(),
                                                    "icon": icons.get(category.lower(), "🧠"),
                                                    "text": value.strip()
                                                })
                                            except Exception:
                                                pass

                                    # Save clean text to in-session history
                                    if clean_text:
                                        state["history"].append({
                                            "role": "vera",
                                            "text": clean_text
                                        })
                                        if len(state["history"]) > MAX_HISTORY * 2:
                                            state["history"] = state["history"][-(MAX_HISTORY * 2):]
                                        logger.info(f"✅ Turn done. History={len(state['history'])}")

                                    await send_ws({"type": "turn_complete"})

                                if getattr(sc, "interrupted", False):
                                    state["response_buffer"] = []
                                    await send_ws({"type": "interrupted"})
                                    logger.info("⚡ Interrupted")

                        except websockets.exceptions.ConnectionClosedError as exc:
                            logger.warning(f"Gemini closed: {exc}")
                        except Exception as exc:
                            logger.error(f"receive_from_gemini: {exc}", exc_info=True)

                    t_send    = asyncio.create_task(send_to_gemini(),      name="send")
                    t_receive = asyncio.create_task(receive_from_gemini(), name="receive")

                    done, pending = await asyncio.wait(
                        [t_send, t_receive],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    if state["frontend_gone"]:
                        logger.info("Frontend gone — stopping")
                        return

                    logger.info("♻️  Reconnecting in 0.1s...")
                    await asyncio.sleep(0.1)

            except Exception as exc:
                logger.error(f"Session error: {exc}", exc_info=True)
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Max retries ({max_retries}) reached. Stopping.")
                    try:
                        await send_ws({"type": "error", "message": "VERA connection failed after max retries."})
                    except Exception:
                        pass
                    return
                backoff = min(2 ** retry_count, 30) * (0.5 + random.random())  # jitter
                logger.info(f"Retrying in {backoff:.1f}s (attempt {retry_count}/{max_retries})...")
                # Only notify frontend on first failure — avoids error flooding
                if retry_count == 1:
                    try:
                        await send_ws({"type": "error", "message": "Connection interrupted — reconnecting..."})
                    except Exception:
                        return
                await asyncio.sleep(backoff)
            else:
                retry_count = 0  # reset on successful session

    try:
        await asyncio.gather(read_from_frontend(), gemini_session_loop(), ws_writer())
    except Exception as exc:
        logger.error(f"Top-level error: {exc}", exc_info=True)
    finally:
        logger.info("🧹 Cleaned up")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "VERA is alive", "model": MODEL}

@app.get("/memories")
async def get_all_memories():
    return load_memory_store()

@app.delete("/memories/{category}")
async def clear_memory_category(category: str):
    if category not in MEMORY_CATEGORIES and category != "all":
        return {"error": f"Use one of: {MEMORY_CATEGORIES} or 'all'"}
    store = load_memory_store()
    if category == "all":
        store = {k: [] for k in MEMORY_CATEGORIES}
    else:
        store[category] = []
    _write_store(store)
    return {"cleared": category}