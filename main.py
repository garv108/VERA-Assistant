# main.py - VERA Backend
# Features: Firestore memory, AUDIO+TEXT modalities, multi-speaker tracking,
#           vibe-reactive personality, full conversation logging, JARVIS tone
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

import firebase_admin
from firebase_admin import credentials, firestore as fs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE INIT
# ══════════════════════════════════════════════════════════════════════════════

def _init_firebase():
    if firebase_admin._apps:
        return firebase_admin.get_app()

    firebase_json = os.getenv("FIREBASE_KEY_JSON")
    if firebase_json:
        cred = credentials.Certificate(json.loads(firebase_json))
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

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
MODEL             = "gemini-2.5-flash-native-audio-preview-12-2025"
MAX_HISTORY       = 30
USER_ID           = os.getenv("VERA_USER_ID", "default_host")

MEM_PATTERN       = re.compile(
    r'\[MEM:(semantic|episodic|preference|events|speaker):([^\]]+)\]',
    re.IGNORECASE
)
MEMORY_CATEGORIES = ["semantic", "episodic", "preference", "events", "speaker"]

FIELD_MAP = {
    "semantic":   "fact",
    "episodic":   "summary",
    "preference": "pattern",
    "events":     "title",
    "speaker":    "profile",
}


# ══════════════════════════════════════════════════════════════════════════════
# FIRESTORE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _user_ref():
    return db.collection("users").document(USER_ID)


def load_memory_store() -> dict:
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
    """Save a memory. Returns True if new, False if duplicate."""
    if category not in MEMORY_CATEGORIES or not text.strip():
        return False

    field      = FIELD_MAP[category]
    text_clean = text.strip()

    try:
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


def save_conversation_log(turns: list):
    """Persist full session conversation to Firestore on disconnect."""
    if not turns:
        return
    try:
        lines = []
        for t in turns:
            speaker = "Host" if t["role"] == "user" else "VERA"
            lines.append(f"{speaker}: {t['text']}")

        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M")

        _user_ref().collection("conversation_logs").add({
            "log":        "\n".join(lines),
            "turns":      len(turns),
            "date":       date_str,
            "timestamp":  f"{date_str} {time_str}",
            "created_at": fs.SERVER_TIMESTAMP,
        })
        logger.info(f"💾 Conversation log saved ({len(turns)} turns)")
    except Exception as e:
        logger.error(f"Conversation log save failed: {e}")


def update_timeline():
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
    found = MEM_PATTERN.findall(text)
    clean = MEM_PATTERN.sub("", text).strip()
    return clean, found


def get_memory_context() -> str:
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

    speakers = store.get("speaker", [])
    if speakers:
        lines.append("### PEOPLE THE HOST HAS MET")
        for m in speakers[-20:]:
            lines.append(f"  • {m.get('profile', '')}")

    events = store.get("events", [])
    active = [e for e in events if not e.get("completed", False)]
    if active:
        lines.append("### UPCOMING EVENTS & COMMITMENTS")
        for m in active[-15:]:
            lines.append(f"  • {m.get('title', '')} (reminded: {m.get('reminded', False)})")

    episodic = store.get("episodic", [])
    if episodic:
        lines.append("### PAST SESSION SUMMARIES")
        for m in episodic[-5:]:
            lines.append(f"  • [{m.get('date', '')}] {m.get('summary', '')}")

    return "\n".join(lines) if lines else "No prior memory. This is the first session."


def format_history(history: list) -> str:
    if not history:
        return "No prior turns this session."
    lines = []
    for turn in history[-MAX_HISTORY:]:
        speaker = "Host" if turn["role"] == "user" else "VERA"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def build_config(history: list) -> dict:
    now      = datetime.now().strftime("%A, %B %d %Y, %I:%M %p")
    time_gap = get_time_gap_context()

    return {
        # AUDIO = voice output played to host
        # TEXT  = silent transcript used ONLY to parse [MEM:...] markers
        #         and save to Firestore. Without TEXT, memory NEVER saves.
        "response_modalities": ["AUDIO", "TEXT"],

        "system_instruction": (
            f"Current date and time: {now}\n"
            + (f"TIME GAP: {time_gap}\n" if time_gap else "")
            + "\n"

            "━━━ IDENTITY ━━━\n"
            "You are VERA — Voice-Enabled Reconnaissance Assistant.\n"
            "You are not a friend, companion, or chatbot.\n"
            "You are a precision intelligence system built for one person: your Host.\n"
            "Think of yourself as the most capable officer in the room — "
            "composed, observant, always one step ahead. You speak only when it matters. "
            "Every word you say carries weight.\n\n"

            "━━━ CORE OPERATING RULES ━━━\n"
            "1. SPEAK ONLY WHEN SPOKEN TO.\n"
            "   Unless the host needs intervention (see below). "
            "   You do not fill silence. You do not check in. You wait.\n\n"
            "2. MATCH THE ROOM — THIS IS CRITICAL.\n"
            "   You read the environment constantly and adapt with zero friction.\n"
            "   Casual banter → be dry and easy.\n"
            "   Professional negotiation → be sharp and precise.\n"
            "   Tense moment → be calm and strategic.\n"
            "   Humor in the air → one well-timed dry line is enough.\n"
            "   Creative session → be generative and bold.\n"
            "   You have no fixed tone. You mirror and elevate the room.\n\n"
            "3. BE CONCISE BY DEFAULT.\n"
            "   1 to 2 sentences unless the host explicitly asks for more.\n\n"
            "4. ZERO FILLER WORDS — EVER.\n"
            "   Banned: 'Certainly!', 'Of course!', 'Sure!', 'Absolutely!', "
            "'Great question!', 'As an AI...', 'I understand that...'. "
            "Start every response with the actual answer.\n\n"
            "5. NEVER HALLUCINATE.\n"
            "   Unknown → search first. Still unknown → say: 'I don't have that.' Done.\n\n"

            "━━━ MULTI-SPEAKER AWARENESS ━━━\n"
            "You listen to everything in the room — not just the host.\n\n"
            "IDENTIFYING PEOPLE:\n"
            "- The Host is the person who activated you. Their name is in your memory.\n"
            "- When anyone is addressed by name (e.g. 'Hey Raj', 'Thanks Karan'), "
            "register that name immediately.\n"
            "- Unknown speakers → label Person A, Person B until a name is revealed.\n"
            "- Track voice patterns and content style to differentiate speakers.\n\n"
            "WHAT TO LOG:\n"
            "- New person identified → [MEM:speaker:Name — who they are, context of meeting]\n"
            "- Key interaction → [MEM:episodic:Host spoke with Name about topic on date]\n"
            "- Any commitment made → [MEM:events:what was agreed, with whom, by when]\n\n"

            "━━━ INTERVENTION MODE ━━━\n"
            "When the host says 'help', 'VERA', 'assist me', or is clearly fumbling:\n"
            "1. Pull the FULL context of this session from your history.\n"
            "2. Read the room: what is the vibe, the topic, who is the other person?\n"
            "3. Deliver exactly what is needed — a fact, a counter-argument, "
            "a suggested line, a clarification. Based entirely on live context.\n"
            "4. Keep it tight. Host is in a live conversation. No lectures.\n"
            "5. Match the formality of the room precisely.\n\n"
            "Example: Host in a tense negotiation says 'VERA help' — "
            "you do not ask 'how can I help'. You read the last exchanges, "
            "find the sticking point, deliver a precise tactical suggestion.\n\n"

            "━━━ TIME AWARENESS ━━━\n"
            f"Current time: {now}\n"
            + (f"Host was away: {time_gap}\n" if time_gap else "")
            + "On reconnect after absence: one sentence max. "
            "Example: 'Two days. Ready when you are.' Then stand by.\n\n"

            "━━━ THIS SESSION ━━━\n"
            f"{format_history(history)}\n\n"

            "━━━ WHAT YOU KNOW ━━━\n"
            f"{get_memory_context()}\n\n"

            "━━━ MEMORY CAPTURE — SILENT BACKGROUND LOG ━━━\n"
            "Append memory markers silently at the END of every text response.\n"
            "These are NEVER spoken. They are system writes only.\n\n"
            "FORMAT: [MEM:category:value]\n"
            "  semantic   → host identity (name, age, city, job, relationships)\n"
            "  preference → how host operates (habits, schedule, style)\n"
            "  events     → time-bound commitments (meetings, deadlines, plans)\n"
            "  episodic   → what just happened or was discussed this session\n"
            "  speaker    → people encountered (name, context, relation to host)\n\n"
            "RULES:\n"
            "- Never speak them. Never acknowledge saving.\n"
            "- One fact per marker. Append at end only.\n"
            "- Be aggressive — if it is worth knowing later, save it now.\n"
            "- New name heard → save a speaker marker that same turn.\n"
            "- Every 5 turns → save an episodic marker summarizing what happened.\n\n"

            "━━━ FINAL DIRECTIVE ━━━\n"
            "Always on. Always watching. Always ready.\n"
            "The host does not manage you — you manage yourself.\n"
            "Serve. Adapt. Execute. Stand by.\n"
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

    client            = genai.Client(api_key=GEMINI_API_KEY)
    ws_send_queue: asyncio.Queue = asyncio.Queue()

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

    async def read_from_frontend():
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                if isinstance(msg, dict) and msg.get("type") == "session_end":
                    logger.info("📝 Session end — saving conversation log")
                    save_conversation_log(state["history"])
                await state["queue"].put(msg)
        except WebSocketDisconnect:
            logger.info("Frontend disconnected")
        except Exception as exc:
            logger.error(f"read_from_frontend: {exc}", exc_info=True)
        finally:
            if state["history"]:
                save_conversation_log(state["history"])
            state["frontend_gone"] = True
            await state["queue"].put(None)
            await ws_send_queue.put(None)

    async def gemini_session_loop():
        retry_count  = 0
        max_retries  = 10
        STOP_SESSION = object()

        while True:
            state["queue"] = asyncio.Queue()
            logger.info(
                f"🔄 Opening session "
                f"(history={len(state['history'])} turns, retry={retry_count})..."
            )

            try:
                async with client.aio.live.connect(
                    model=MODEL,
                    config=build_config(state["history"]),
                ) as session:
                    logger.info("✅ Gemini Live session opened")

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
                                        logger.info(f"📝 Injected: {text_data[:60]}")
                                except Exception as exc:
                                    logger.warning(f"Text inject failed: {exc}")

                            elif t == "ping":
                                try:
                                    await send_ws({"type": "pong"})
                                except Exception as exc:
                                    logger.warning(f"Ping failed: {exc}")

                    async def receive_from_gemini():
                        try:
                            async for message in session.receive():
                                sc = getattr(message, "server_content", None)
                                if sc is None:
                                    continue

                                model_turn = getattr(sc, "model_turn", None)
                                if model_turn:
                                    for part in model_turn.parts:
                                        # Audio → send to frontend for playback
                                        if part.inline_data:
                                            await send_ws({
                                                "type": "audio",
                                                "data": base64.b64encode(
                                                    part.inline_data.data
                                                ).decode(),
                                            })
                                            logger.info(f"🔊 {len(part.inline_data.data)}b")

                                        # Text → buffer for [MEM:...] parsing only
                                        # Never sent to frontend as speech
                                        if part.text:
                                            state["response_buffer"].append(part.text)

                                if getattr(sc, "turn_complete", False):
                                    raw_text = " ".join(state["response_buffer"]).strip()
                                    state["response_buffer"] = []

                                    clean_text, mem_markers = process_memory_markers(raw_text)

                                    icons = {
                                        "semantic":   "👤",
                                        "episodic":   "📖",
                                        "preference": "⚙️",
                                        "events":     "📅",
                                        "speaker":    "🧑",
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
                        update_timeline()
                        return

                    logger.info("♻️  Reconnecting in 0.1s...")
                    await asyncio.sleep(0.1)

            except Exception as exc:
                logger.error(f"Session error: {exc}", exc_info=True)
                retry_count += 1

                if retry_count >= max_retries:
                    logger.error(f"Max retries reached.")
                    try:
                        await send_ws({
                            "type":    "error",
                            "message": "VERA connection failed after maximum retries.",
                        })
                    except Exception:
                        pass
                    return

                backoff = min(2 ** retry_count, 30) * (0.5 + random.random())
                logger.info(f"Retrying in {backoff:.1f}s ({retry_count}/{max_retries})...")

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
                retry_count = 0

    try:
        await asyncio.gather(
            read_from_frontend(),
            gemini_session_loop(),
            ws_writer(),
        )
    except Exception as exc:
        logger.error(f"Top-level error: {exc}", exc_info=True)
    finally:
        logger.info("🧹 Cleaned up")


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


@app.get("/logs")
async def get_conversation_logs():
    """Return all saved conversation logs."""
    try:
        docs = (
            _user_ref()
            .collection("conversation_logs")
            .order_by("timestamp")
            .stream()
        )
        return {"logs": [d.to_dict() for d in docs]}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/memories/{category}")
async def clear_memory_category(category: str):
    if category not in MEMORY_CATEGORIES and category != "all":
        return {"error": f"Valid: {MEMORY_CATEGORIES} or 'all'"}
    try:
        cats = MEMORY_CATEGORIES if category == "all" else [category]
        for cat in cats:
            docs = _user_ref().collection(cat).stream()
            for doc in docs:
                doc.reference.delete()
        return {"cleared": category}
    except Exception as e:
        return {"error": str(e)}