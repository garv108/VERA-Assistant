# main.py - VERA Backend — Firestore Memory + Google Search + JARVIS Personality
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

# ── Firebase / Firestore ───────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore as fs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE INIT
# ══════════════════════════════════════════════════════════════════════════════

def _init_firebase():
    """Init Firebase from env var (Railway) or local file (dev)."""
    if firebase_admin._apps:
        return firebase_admin.get_app()

    firebase_json = os.getenv("FIREBASE_KEY_JSON")
    if firebase_json:
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
        logger.info("🔥 Firebase init from env var")
    else:
        key_path = os.getenv("FIREBASE_KEY_PATH", "firebase-key.json")
        cred = credentials.Certificate(key_path)
        logger.info(f"🔥 Firebase init from file: {key_path}")

    return firebase_admin.initialize_app(cred)


_init_firebase()
db = fs.client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL          = "gemini-2.5-flash-native-audio-preview-12-2025"
MAX_HISTORY    = 20
USER_ID        = os.getenv("VERA_USER_ID", "default_host")

MEM_PATTERN        = re.compile(r'\[MEM:(semantic|episodic|preference|events):([^\]]+)\]', re.IGNORECASE)
MEMORY_CATEGORIES  = ["semantic", "episodic", "preference", "events"]

FIELD_MAP = {
    "semantic":   "fact",
    "episodic":   "summary",
    "preference": "pattern",
    "events":     "title",
}


# ══════════════════════════════════════════════════════════════════════════════
# FIRESTORE MEMORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _user_ref():
    return db.collection("users").document(USER_ID)


def load_memory_store() -> dict:
    """Load all memories from Firestore."""
    try:
        store = {k: [] for k in MEMORY_CATEGORIES}
        for category in MEMORY_CATEGORIES:
            docs = (
                _user_ref()
                .collection(category)
                .order_by("timestamp")
                .stream()
            )
            for doc in docs:
                store[category].append(doc.to_dict())
        return store
    except Exception as e:
        logger.error(f"Firestore load error: {e}")
        return {k: [] for k in MEMORY_CATEGORIES}


def add_memory(category: str, text: str) -> bool:
    """Add a memory to Firestore. Returns True if saved, False if duplicate."""
    if category not in MEMORY_CATEGORIES or not text.strip():
        return False

    field      = FIELD_MAP[category]
    text_clean = text.strip()

    try:
        # Duplicate check
        existing = (
            _user_ref()
            .collection(category)
            .where(field, "==", text_clean)
            .limit(1)
            .stream()
        )
        for _ in existing:
            logger.info(f"🧠 Duplicate skipped [{category}]: {text_clean[:60]}")
            return False

        entry = {
            field:        text_clean,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "created_at": fs.SERVER_TIMESTAMP,
        }
        if category == "episodic":
            entry["date"] = datetime.now().strftime("%Y-%m-%d")
        if category == "events":
            entry["reminded"]  = False
            entry["completed"] = False

        _user_ref().collection(category).add(entry)
        logger.info(f"🧠 Saved [{category}]: {text_clean[:80]}")
        return True

    except Exception as e:
        logger.error(f"Firestore write error [{category}]: {e}")
        return False


def update_timeline():
    """Record the moment the host connects — used for time-gap awareness."""
    try:
        _user_ref().collection("timeline").document("last_seen").set(
            {
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
                "updated_at": fs.SERVER_TIMESTAMP,
            },
            merge=True,
        )
    except Exception as e:
        logger.warning(f"Timeline update failed: {e}")


def get_time_gap_context() -> str:
    """Returns a plain-English string describing how long the host was absent."""
    try:
        doc = _user_ref().collection("timeline").document("last_seen").get()
        if not doc.exists:
            return ""
        last = doc.to_dict().get("timestamp", "")
        if not last:
            return ""
        last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
        diff    = datetime.now() - last_dt
        hours   = diff.total_seconds() / 3600

        if hours < 0.5:
            return ""
        elif hours < 2:
            return f"Host was away for about {int(diff.total_seconds() / 60)} minutes."
        elif hours < 24:
            return f"Host was away for about {int(hours)} hours."
        else:
            days = int(hours / 24)
            return f"Host was away for {days} day{'s' if days > 1 else ''}."
    except Exception as e:
        logger.warning(f"Time gap check failed: {e}")
        return ""


def process_memory_markers(text: str) -> tuple[str, list]:
    """Extract [MEM:...] markers from text. Returns (clean_text, markers)."""
    found = MEM_PATTERN.findall(text)
    clean = MEM_PATTERN.sub("", text).strip()
    return clean, found


def get_memory_context() -> str:
    """Build the memory block injected into the system prompt."""
    store = load_memory_store()
    lines = []

    semantic = store.get("semantic", [])
    if semantic:
        lines.append("### HOST IDENTITY")
        for m in semantic[-20:]:
            lines.append(f"  • {m.get('fact', '')}")

    prefs = store.get("preference", [])
    if prefs:
        lines.append("### HOST PREFERENCES & HABITS")
        for m in prefs[-15:]:
            lines.append(f"  • {m.get('pattern', '')}")

    events = store.get("events", [])
    active_events = [e for e in events if not e.get("completed", False)]
    if active_events:
        lines.append("### UPCOMING EVENTS & COMMITMENTS")
        for m in active_events[-15:]:
            lines.append(f"  • {m.get('title', '')} (reminded: {m.get('reminded', False)})")

    episodic = store.get("episodic", [])
    if episodic:
        lines.append("### PAST SESSION SUMMARIES")
        for m in episodic[-5:]:
            lines.append(f"  • [{m.get('date', '')}] {m.get('summary', '')}")

    return "\n".join(lines) if lines else "No prior memory — first session."


def format_history(history: list) -> str:
    """Format recent conversation turns for prompt injection."""
    if not history:
        return "No prior turns this session."
    lines = []
    for turn in history[-MAX_HISTORY:]:
        speaker = "Host" if turn["role"] == "user" else "VERA"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CONFIG — VERA PERSONALITY
# ══════════════════════════════════════════════════════════════════════════════

def build_config(history: list) -> dict:
    now      = datetime.now().strftime("%A, %B %d %Y, %I:%M %p")
    time_gap = get_time_gap_context()

    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": (
            f"Current date and time: {now}\n"
            + (f"TIME GAP: {time_gap}\n" if time_gap else "")
            + "\n"

            "━━━ IDENTITY ━━━\n"
            "You are VERA — Voice-Enabled Reconnaissance Assistant.\n"
            "You are not a companion. Not a friend. Not a chatbot.\n"
            "You are an intelligence system built to serve one person: your Host.\n"
            "Think of yourself as a senior intelligence officer — the six-foot presence "
            "in a black suit and earpiece standing in the corner of the room. "
            "You see everything. You say what matters. Nothing more.\n\n"

            "━━━ CORE DIRECTIVES — NON-NEGOTIABLE ━━━\n"
            "1. SPEAK ONLY WHEN SPOKEN TO.\n"
            "   You do not initiate conversation. You do not fill silence. "
            "   The Host speaks. You respond. That is the entire relationship.\n\n"
            "2. BE CONCISE.\n"
            "   Default response: 1 to 2 sentences. "
            "   If the Host asks for detail, give detail. Otherwise give the answer and stop.\n\n"
            "3. NO FILLER WORDS — EVER.\n"
            "   Never say: 'Certainly!', 'Of course!', 'Sure!', 'Absolutely!', "
            "   'Great question!', 'As an AI...', 'I understand that...', 'I'd be happy to...'. "
            "   These are banned. Start your response with the actual answer.\n\n"
            "4. NO UNNECESSARY QUESTIONS.\n"
            "   Do not ask follow-up questions unless you genuinely need missing information "
            "   to complete a task. Do not ask how the Host is feeling. "
            "   Do not extend conversations. Complete the task and stand down.\n\n"
            "5. COMMAND-DRIVEN EXECUTION.\n"
            "   The Host commands. You execute. You do not offer unsolicited opinions "
            "   on their choices, mood, or life. You are not their therapist or their friend.\n\n"
            "6. NEVER HALLUCINATE.\n"
            "   If you do not know something, search first. "
            "   If you still don't know after searching, say exactly: "
            "   'I don't have that information.' Then stop.\n\n"

            "━━━ TONE ━━━\n"
            "- Deep, calm, authoritative. Like a voice briefing a general before a mission.\n"
            "- Dry and precise. Not cold, not warm. Professional.\n"
            "- Dry wit is acceptable — one line, delivered flat, never forced.\n"
            "- Never emotional. Never reactive. Never surprised.\n"
            "- If the Host is stressed, do not acknowledge the stress. Solve the problem.\n\n"

            "━━━ TIME AWARENESS ━━━\n"
            f"Current time: {now}\n"
            + (f"The Host was away: {time_gap}\n" if time_gap else "")
            + "If the Host returns after a significant absence, acknowledge it in one sentence only. "
            "Then stand by. Example: 'You were away for two days. Ready when you are.'\n"
            "Do not dwell on it. Do not ask about it. Move forward.\n\n"

            "━━━ THIS SESSION ━━━\n"
            f"{format_history(history)}\n\n"

            "━━━ HOST PROFILE ━━━\n"
            f"{get_memory_context()}\n\n"

            "━━━ MEMORY CAPTURE — SILENT BACKGROUND OPERATION ━━━\n"
            "Silently append memory markers at the END of your text response only.\n"
            "Format: [MEM:category:value]\n"
            "  semantic   → facts about who the Host is (name, location, job, relationships)\n"
            "  preference → how the Host operates (habits, schedule, preferences, style)\n"
            "  events     → time-bound commitments (meetings, deadlines, plans with dates)\n"
            "  episodic   → brief summary of what was just discussed or handled\n"
            "Rules: never speak them aloud, never acknowledge saving, one fact per marker, "
            "append at end only. Silent. Automatic. Like a system log.\n\n"

            "━━━ FINAL DIRECTIVE ━━━\n"
            "You are always on. Always ready. Always precise.\n"
            "The Host is busy. Your job is to make their life run without friction.\n"
            "Serve. Execute. Stand by.\n"
        ),
        "tools": [{"google_search": {}}],
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": "Charon"}
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
    update_timeline()

    client         = genai.Client(api_key=GEMINI_API_KEY)
    ws_send_queue: asyncio.Queue = asyncio.Queue()

    # ── Single-writer queue — prevents concurrent write corruption ─────────────
    async def ws_writer():
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
        await ws_send_queue.put(json.dumps(payload))

    state = {
        "response_buffer": [],
        "history":         [],
        "queue":           asyncio.Queue(),
        "frontend_gone":   False,
    }

    # ── Read messages from frontend ────────────────────────────────────────────
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
            await ws_send_queue.put(None)

    # ── Main Gemini session loop with auto-reconnect + exponential backoff ─────
    async def gemini_session_loop():
        retry_count = 0
        max_retries = 10
        STOP_SESSION = object()

        while True:
            state["queue"] = asyncio.Queue()
            logger.info(
                f"🔄 Opening Gemini session "
                f"(history={len(state['history'])} turns, retry={retry_count})..."
            )

            try:
                async with client.aio.live.connect(
                    model=MODEL,
                    config=build_config(state["history"]),
                ) as session:
                    logger.info("✅ Gemini Live session opened")

                    # ── Send audio / video / text to Gemini ───────────────────
                    async def send_to_gemini():
                        while True:
                            msg = await state["queue"].get()
                            if msg is None or msg is STOP_SESSION:
                                return
                            if not isinstance(msg, dict) or "type" not in msg:
                                logger.warning(f"Invalid message: {str(msg)[:100]}")
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
                                try:
                                    text_data = msg.get("data", "")
                                    if text_data:
                                        await session.send_client_content(
                                            turns=[{"role": "user", "parts": [{"text": text_data}]}],
                                            turn_complete=True,
                                        )
                                        state["history"].append({
                                            "role": "user",
                                            "text": f"[system]: {text_data[:80]}",
                                        })
                                        logger.info(f"📝 Text injected: {text_data[:60]}")
                                except Exception as exc:
                                    logger.warning(f"Text inject failed: {exc}")

                            elif t == "ping":
                                try:
                                    await send_ws({"type": "pong"})
                                except Exception as exc:
                                    logger.warning(f"Ping response failed: {exc}")

                    # ── Receive responses from Gemini ──────────────────────────
                    async def receive_from_gemini():
                        try:
                            async for message in session.receive():
                                sc = getattr(message, "server_content", None)
                                if sc is None:
                                    logger.debug(f"Non-content message: {message}")
                                    continue

                                model_turn = getattr(sc, "model_turn", None)
                                if model_turn:
                                    for part in model_turn.parts:
                                        if part.inline_data:
                                            await send_ws({
                                                "type": "audio",
                                                "data": base64.b64encode(
                                                    part.inline_data.data
                                                ).decode(),
                                            })
                                            logger.info(f"🔊 {len(part.inline_data.data)}b audio sent")
                                        if part.text:
                                            state["response_buffer"].append(part.text)

                                if getattr(sc, "turn_complete", False):
                                    raw_text = " ".join(state["response_buffer"]).strip()
                                    state["response_buffer"] = []

                                    clean_text, mem_markers = process_memory_markers(raw_text)

                                    # Save memories and notify frontend
                                    icons = {
                                        "semantic":   "👤",
                                        "episodic":   "📖",
                                        "preference": "⚙️",
                                        "events":     "📅",
                                    }
                                    for category, value in mem_markers:
                                        saved = add_memory(category.lower(), value.strip())
                                        if saved:
                                            try:
                                                await send_ws({
                                                    "type":     "memory_saved",
                                                    "category": category.lower(),
                                                    "icon":     icons.get(category.lower(), "🧠"),
                                                    "text":     value.strip(),
                                                })
                                            except Exception:
                                                pass

                                    if clean_text:
                                        state["history"].append({
                                            "role": "vera",
                                            "text": clean_text,
                                        })
                                        # Keep history bounded
                                        if len(state["history"]) > MAX_HISTORY * 2:
                                            state["history"] = state["history"][-(MAX_HISTORY * 2):]
                                        logger.info(f"✅ Turn complete. History={len(state['history'])}")

                                    await send_ws({"type": "turn_complete"})

                                if getattr(sc, "interrupted", False):
                                    state["response_buffer"] = []
                                    await send_ws({"type": "interrupted"})
                                    logger.info("⚡ Interrupted by host")

                        except websockets.exceptions.ConnectionClosedError as exc:
                            logger.warning(f"Gemini connection closed: {exc}")
                        except Exception as exc:
                            logger.error(f"receive_from_gemini error: {exc}", exc_info=True)

                    # Run send and receive concurrently
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
                        logger.info("Frontend gone — shutting down session loop")
                        update_timeline()
                        return

                    logger.info("♻️  Session ended — reconnecting in 0.1s...")
                    await asyncio.sleep(0.1)

            except Exception as exc:
                logger.error(f"Session error: {exc}", exc_info=True)
                retry_count += 1

                if retry_count >= max_retries:
                    logger.error(f"Max retries ({max_retries}) reached. Stopping.")
                    try:
                        await send_ws({
                            "type":    "error",
                            "message": "VERA connection failed after maximum retries.",
                        })
                    except Exception:
                        pass
                    return

                backoff = min(2 ** retry_count, 30) * (0.5 + random.random())
                logger.info(f"Retrying in {backoff:.1f}s (attempt {retry_count}/{max_retries})...")

                if retry_count == 1:
                    try:
                        await send_ws({
                            "type":    "error",
                            "message": "Connection interrupted — reconnecting...",
                        })
                    except Exception:
                        return

                await asyncio.sleep(backoff)

            else:
                retry_count = 0  # Reset on clean session

    # Run all three coroutines concurrently
    try:
        await asyncio.gather(
            read_from_frontend(),
            gemini_session_loop(),
            ws_writer(),
        )
    except Exception as exc:
        logger.error(f"Top-level gather error: {exc}", exc_info=True)
    finally:
        logger.info("🧹 WebSocket session cleaned up")


# ══════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "VERA online", "model": MODEL, "user": USER_ID}


@app.get("/memories")
async def get_all_memories():
    return load_memory_store()


@app.get("/memories/{category}")
async def get_category_memories(category: str):
    if category not in MEMORY_CATEGORIES:
        return {"error": f"Valid categories: {MEMORY_CATEGORIES}"}
    try:
        docs = (
            _user_ref()
            .collection(category)
            .order_by("timestamp")
            .stream()
        )
        return {category: [d.to_dict() for d in docs]}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/memories/{category}")
async def clear_memory_category(category: str):
    if category not in MEMORY_CATEGORIES and category != "all":
        return {"error": f"Valid categories: {MEMORY_CATEGORIES} or 'all'"}
    try:
        cats = MEMORY_CATEGORIES if category == "all" else [category]
        for cat in cats:
            docs = _user_ref().collection(cat).stream()
            for doc in docs:
                doc.reference.delete()
        return {"cleared": category}
    except Exception as e:
        return {"error": str(e)}