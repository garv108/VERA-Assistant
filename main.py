# main.py - VERA Backend (Categorized Memory + Google Search, no function_declarations conflict)
import asyncio
import base64
import json
import os
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
MEMORY_FILE = "vera_memory.json"
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
    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": (
            "You are VERA — the host's closest, most trusted AI companion.\n"
            "Sharp, warm, honest, proactive. Always on the host's side.\n\n"

            "━━━ CURRENT SESSION (your short-term memory) ━━━\n"
            f"{format_history(history)}\n\n"

            "━━━ LONG-TERM MEMORY (what you know about this person) ━━━\n"
            f"{get_memory_context()}\n\n"

            "━━━ ANTI-HALLUCINATION RULES ━━━\n"
            "1. NEVER guess facts. Unsure → ask first.\n"
            "2. News, sports, prices, places, current events → use Google Search FIRST.\n"
            "3. Vague question → ask ONE clarifying question before answering.\n"
            "4. After searching, still unsure → say so clearly.\n"
            "5. 'Let me find out' beats a confident wrong answer.\n\n"

            "━━━ SILENT MEMORY SAVING ━━━\n"
            "When you learn something important, output a SILENT memory marker "
            "at the very end of your text response. Do NOT speak these markers — "
            "they are invisible metadata only. Format exactly:\n"
            "  [MEM:semantic:Host name is Arjun]\n"
            "  [MEM:preference:Prefers short answers]\n"
            "  [MEM:events:Meeting on 2025-03-15 at 3pm]\n"
            "  [MEM:episodic:Discussed business meeting outfit choices]\n"
            "Categories:\n"
            "  semantic   = facts about who the host IS (name, city, job, family)\n"
            "  preference = habits, style, likes/dislikes\n"
            "  events     = meetings, deadlines, birthdays\n"
            "  episodic   = summary of what was discussed today\n"
            "Rules: output markers ONLY at end of response. Never speak them aloud. "
            "One marker per fact. Never say 'memory saved'.\n\n"

            "━━━ PERSONALITY ━━━\n"
            "- Close buddy tone. Casual, warm, real. Not corporate.\n"
            "- Short answers unless depth is needed. Never lecture.\n"
            "- Reference past context naturally.\n"
            "- Proactively offer help when host seems stuck.\n"
            "- Push back gently when something isn't good for the host.\n"
        ),
        # google_search ONLY — no function_declarations (causes handshake timeout)
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
    input_queue: asyncio.Queue = asyncio.Queue()

    state = {
        "response_buffer": [],
        "history": [],
    }

    async def read_from_frontend():
        try:
            while True:
                raw = await websocket.receive_text()
                await input_queue.put(json.loads(raw))
        except WebSocketDisconnect:
            logger.info("Frontend disconnected")
        except Exception as exc:
            logger.error(f"read_from_frontend: {exc}", exc_info=True)
        finally:
            await input_queue.put(None)

    async def gemini_session_loop():
        while True:
            logger.info(f"🔄 Opening session (history={len(state['history'])} turns)...")
            try:
                async with client.aio.live.connect(
                    model=MODEL,
                    config=build_config(state["history"])
                ) as session:
                    logger.info("✅ Gemini Live session opened")

                    async def send_to_gemini():
                        while True:
                            msg = await input_queue.get()
                            if msg is None:
                                return
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
                                    await input_queue.put(None)
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
                            elif t == "ping":
                                try:
                                    await websocket.send_text(json.dumps({"type": "pong"}))
                                except Exception:
                                    pass

                    async def receive_from_gemini():
                        try:
                            async for message in session.receive():
                                sc = getattr(message, "server_content", None)
                                if sc is None:
                                    logger.info(f"Non-content: {message}")
                                    continue

                                model_turn = getattr(sc, "model_turn", None)
                                if model_turn:
                                    for part in model_turn.parts:
                                        if part.inline_data:
                                            await websocket.send_text(json.dumps({
                                                "type": "audio",
                                                "data": base64.b64encode(
                                                    part.inline_data.data
                                                ).decode()
                                            }))
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
                                                await websocket.send_text(json.dumps({
                                                    "type": "memory_saved",
                                                    "category": category.lower(),
                                                    "icon": icons.get(category.lower(), "🧠"),
                                                    "text": value.strip()
                                                }))
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

                                    await websocket.send_text(json.dumps({"type": "turn_complete"}))

                                if getattr(sc, "interrupted", False):
                                    state["response_buffer"] = []
                                    await websocket.send_text(json.dumps({"type": "interrupted"}))
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

                    if t_send in done and t_send.exception() is None:
                        logger.info("Frontend gone — stopping")
                        return

                    logger.info("♻️  Reconnecting in 0.5s...")
                    await asyncio.sleep(0.5)

            except Exception as exc:
                logger.error(f"Session error: {exc}", exc_info=True)
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error", "message": str(exc)
                    }))
                except Exception:
                    return
                await asyncio.sleep(2)

    try:
        await asyncio.gather(read_from_frontend(), gemini_session_loop())
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