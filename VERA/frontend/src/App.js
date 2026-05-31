// src/App.js - VERA Frontend — Full Premium Build
// Features: Neural UI, Onboarding, Memory Timeline, Proactive Check-in, Session Recap
import React, { useState, useRef, useEffect, useCallback } from "react";
import MemoryDashboard from "./MemoryDashboard";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";
const wsProtocol = API_URL.startsWith("https") ? "wss" : "ws";
const wsHost = API_URL.replace(/^https?:\/\//, "");
const WS_URL = `${wsProtocol}://${wsHost}/ws/vera`;const AUDIO_SAMPLE_RATE = 16000;
const VIDEO_INTERVAL_MS = 5000;
const FLUSH_INTERVAL_MS = 50;
const PROACTIVE_SILENCE_MS = 120000; // 2 minutes — VERA checks in

const WORKLET_CODE = `
class PCMCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch) this.port.postMessage(ch.slice());
    return true;
  }
}
registerProcessor("pcm-capture", PCMCapture);
`;

// ── Google Font injection ────────────────────────────────────────────────────
const FONT_LINK = document.createElement("link");
FONT_LINK.rel = "stylesheet";
FONT_LINK.href = "https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap";
document.head.appendChild(FONT_LINK);

// ── Keyframe injection ───────────────────────────────────────────────────────
const STYLE = document.createElement("style");
STYLE.textContent = `
  * { box-sizing: border-box; }
  body { margin: 0; background: #030308; }
  @keyframes pulse-ring {
    0%   { transform: scale(0.95); opacity: 0.8; }
    50%  { transform: scale(1.05); opacity: 0.4; }
    100% { transform: scale(0.95); opacity: 0.8; }
  }
  @keyframes orb-glow {
    0%   { box-shadow: 0 0 20px 4px #00d4ff44, 0 0 60px 10px #0066ff22; }
    50%  { box-shadow: 0 0 40px 8px #00d4ff66, 0 0 100px 20px #0066ff44; }
    100% { box-shadow: 0 0 20px 4px #00d4ff44, 0 0 60px 10px #00d4ff22; }
  }
  @keyframes orb-speak {
    0%   { box-shadow: 0 0 30px 6px #00ff9944, 0 0 80px 15px #00cc7722; transform: scale(1); }
    25%  { transform: scale(1.06); box-shadow: 0 0 50px 12px #00ff9966, 0 0 120px 25px #00cc7744; }
    50%  { transform: scale(0.97); }
    75%  { transform: scale(1.04); }
    100% { box-shadow: 0 0 30px 6px #00ff9944, 0 0 80px 15px #00cc7722; transform: scale(1); }
  }
  @keyframes orb-think {
    0%   { box-shadow: 0 0 20px 4px #ff990044, 0 0 60px 10px #ff660022; }
    50%  { box-shadow: 0 0 40px 8px #ff990066, 0 0 100px 20px #ff660044; }
    100% { box-shadow: 0 0 20px 4px #ff990044, 0 0 60px 10px #ff660022; }
  }
  @keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
  }
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes wave {
    0%, 100% { transform: scaleY(0.3); }
    50%       { transform: scaleY(1); }
  }
  @keyframes dot-bounce {
    0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
    40%           { transform: scale(1); opacity: 1; }
  }
  @keyframes mem-slide {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  @keyframes grid-drift {
    0%   { background-position: 0 0; }
    100% { background-position: 40px 40px; }
  }
  .vera-btn {
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
  }
  .vera-btn:hover:not(:disabled) {
    transform: translateY(-2px);
  }
  .vera-btn:active:not(:disabled) {
    transform: translateY(0px);
  }
  .mem-entry { animation: mem-slide 0.3s ease forwards; }
  .log-entry { animation: fadeInUp 0.25s ease forwards; }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #1a2a3a; border-radius: 2px; }
`;
document.head.appendChild(STYLE);

// ── Wave Visualizer ──────────────────────────────────────────────────────────
function WaveVisualizer({ active, color = "#00d4ff" }) {
  const bars = 12;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 3, height: 28 }}>
      {Array.from({ length: bars }).map((_, i) => (
        <div key={i} style={{
          width: 3,
          height: "100%",
          background: color,
          borderRadius: 2,
          opacity: active ? 0.9 : 0.2,
          transformOrigin: "center",
          animation: active ? `wave ${0.6 + (i % 4) * 0.15}s ease-in-out infinite` : "none",
          animationDelay: `${i * 0.07}s`,
        }} />
      ))}
    </div>
  );
}

// ── Thinking Dots ────────────────────────────────────────────────────────────
function ThinkingDots() {
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      {[0, 1, 2].map(i => (
        <div key={i} style={{
          width: 7, height: 7, borderRadius: "50%",
          background: "#ff9900",
          animation: `dot-bounce 1.2s ease-in-out infinite`,
          animationDelay: `${i * 0.2}s`,
        }} />
      ))}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
// Moved outside component — avoids reallocation on every render
const ONBOARD_STEPS = [
  "What's your name?",
  "What city are you in?",
  "What do you do for work?",
  "What's one thing you're working on right now?",
  "How do you prefer I talk to you — formal or casual?"
];

export default function App() {
  const [status, setStatus]         = useState("idle");
  const [page, setPage]             = useState("main"); // "main" | "memory"
  const [log, setLog]               = useState([]);
  const [memories, setMemories]     = useState([]);
  const [memPanel, setMemPanel]     = useState(false);
  const [veraState, setVeraState]   = useState("idle");
  const [isOnboarding, setIsOnboarding] = useState(false);
  const [onboardStep, setOnboardStep]   = useState(0);
  const [sessionTurns, setSessionTurns] = useState(0);
  // storedMemoryCount: fetched from Firestore on mount
  // Used to decide whether to run onboarding (0 = truly first session)
  const [storedMemoryCount, setStoredMemoryCount] = useState(null); // null = loading

  const wsRef             = useRef(null);
  const cameraWatchdogRef = useRef(null); // auto-restarts camera if it dies
  const audioCtxRef       = useRef(null); // capture: 16kHz
  const playCtxRef        = useRef(null); // playback: 24kHz
  const workletNodeRef    = useRef(null);
  const mediaStreamRef    = useRef(null);
  const videoStreamRef    = useRef(null);
  const videoCanvasRef    = useRef(null);
  const videoTimerRef     = useRef(null);
  const flushTimerRef     = useRef(null);
  const proactiveTimerRef = useRef(null);
  const recapTimerRef     = useRef(null);
  const pcmBufferRef      = useRef([]);
  const audioQueueRef     = useRef([]);
  const isPlayingRef      = useRef(false);
  const isStoppingRef     = useRef(false);
  const mountedRef        = useRef(true);
  const logRef            = useRef([]);

  const addLog = useCallback((msg) => {
    const ts = new Date().toLocaleTimeString();
    const entry = `[${ts}] ${msg}`;
    logRef.current = [...logRef.current.slice(-9), entry];
    setLog([...logRef.current]);
  }, []);

  // ── Gapless audio playback via Web Audio timeline ──────────────────────────
  // scheduleNextAt tracks the exact time the next chunk should start
  // so chunks are scheduled back-to-back with zero gap — no stuttering
  const scheduleNextAtRef = useRef(0);
  const activeSourcesRef  = useRef([]);

  const playNext = useCallback(() => {
    const ctx = playCtxRef.current;
    if (!ctx || ctx.state === "closed") return;

    while (audioQueueRef.current.length > 0) {
      const buf = audioQueueRef.current.shift();
      try {
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);

        // Schedule back-to-back with zero gap — gapless playback
        const startAt = Math.max(scheduleNextAtRef.current, ctx.currentTime + 0.01);
        src.start(startAt);
        scheduleNextAtRef.current = startAt + buf.duration;

        activeSourcesRef.current.push(src);
        src.onended = () => {
          activeSourcesRef.current = activeSourcesRef.current.filter(s => s !== src);
          if (activeSourcesRef.current.length === 0 && audioQueueRef.current.length === 0) {
            isPlayingRef.current = false;
            setVeraState("listening");
          }
        };
      } catch (e) {
        console.error("Audio schedule error:", e);
      }
    }
  }, []);

  const handleAudioResponse = useCallback((b64) => {
    // Use dedicated 24kHz playback context — matches Gemini output exactly
    let pCtx = playCtxRef.current;
    if (!pCtx || pCtx.state === "closed") {
      pCtx = new AudioContext({ sampleRate: 24000 });
      playCtxRef.current = pCtx;
    }
    try {
      const raw   = atob(b64);
      const bytes = Uint8Array.from(raw, c => c.charCodeAt(0));
      const pcm16 = new Int16Array(bytes.buffer);
      const f32   = Float32Array.from(pcm16, s => s / 32768);
      const buf   = pCtx.createBuffer(1, f32.length, 24000);
      buf.copyToChannel(f32, 0);
      audioQueueRef.current.push(buf);
      isPlayingRef.current = true;
      setVeraState("speaking");
      if (pCtx.state === "suspended") pCtx.resume().then(playNext).catch(() => {});
      else playNext();
    } catch (e) { console.error("Playback error:", e); }
  }, [playNext]);

  // ── Flush audio ─────────────────────────────────────────────────────────────
  const flushAudio = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!pcmBufferRef.current.length) return;
    const totalLen = pcmBufferRef.current.reduce((s, a) => s + a.length, 0);
    const combined = new Int16Array(totalLen);
    let offset = 0;
    for (const chunk of pcmBufferRef.current) { combined.set(chunk, offset); offset += chunk.length; }
    pcmBufferRef.current = [];
    const bytes  = new Uint8Array(combined.buffer);
    const binary = Array.from(bytes, b => String.fromCharCode(b)).join("");
    ws.send(JSON.stringify({ type: "audio", data: btoa(binary) }));
  }, []);

  // ── Video frame ─────────────────────────────────────────────────────────────
  const sendVideo = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const video  = document.getElementById("vera-video");
    const canvas = videoCanvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;
    canvas.width = 320; canvas.height = 240;
    const ctx2d = canvas.getContext("2d");
    if (!ctx2d) return;
    ctx2d.drawImage(video, 0, 0, 320, 240);
    const frameData = canvas.toDataURL("image/jpeg", 0.6).split(",")[1];
    if (!frameData) return;
    ws.send(JSON.stringify({ type: "video", data: frameData }));
  }, []);

  // ── Send text to VERA (for proactive trigger + onboarding) ──────────────────
  const sendTextToVera = useCallback((text) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "text", data: text }));
  }, []);

  // ── Proactive silence check-in ───────────────────────────────────────────────
  const resetProactiveTimer = useCallback(() => {
    clearTimeout(proactiveTimerRef.current);
    proactiveTimerRef.current = setTimeout(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN && !isPlayingRef.current) {
        addLog("⏰ VERA checking in...");
        sendTextToVera("[PROACTIVE_CHECKIN] The user has been silent for 2 minutes. Proactively check in with them — ask if they need help, or share something interesting related to what they told you before. Keep it short and warm.");
      }
    }, PROACTIVE_SILENCE_MS);
  }, [addLog, sendTextToVera]);

  // ── Cleanup ─────────────────────────────────────────────────────────────────
  const cleanup = useCallback(() => {
    isStoppingRef.current = true;
    clearInterval(flushTimerRef.current);
    clearInterval(videoTimerRef.current);
    clearInterval(cameraWatchdogRef.current);
    clearTimeout(proactiveTimerRef.current);
    clearTimeout(recapTimerRef.current);
    pcmBufferRef.current  = [];
    audioQueueRef.current = [];
    isPlayingRef.current  = false;
    setVeraState("idle");
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    playCtxRef.current?.close();
    playCtxRef.current = null;
    scheduleNextAtRef.current = 0;
    mediaStreamRef.current?.getTracks().forEach(t => t.stop());
    mediaStreamRef.current = null;
    videoStreamRef.current?.getTracks().forEach(t => t.stop());
    videoStreamRef.current = null;
    const videoEl = document.getElementById("vera-video");
    if (videoEl) videoEl.srcObject = null;
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // ── Start VERA ───────────────────────────────────────────────────────────────
  const startVERA = useCallback(async () => {
    const rs = wsRef.current?.readyState;
    if (rs === WebSocket.OPEN || rs === WebSocket.CONNECTING) return;
    if (!mountedRef.current) return;
    cleanup();
    isStoppingRef.current = false;
    setStatus("connecting");
    setVeraState("idle");
    addLog("🔌 Initializing VERA...");

    try {
      const audioStream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: AUDIO_SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        video: false,
      });
      mediaStreamRef.current = audioStream;

      const videoStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 320, height: 240 },
        audio: false,
      });
      videoStreamRef.current = videoStream;
      const videoEl = document.getElementById("vera-video");
      if (videoEl) videoEl.srcObject = videoStream;

      // AudioContext must be created during user gesture
      const ctx = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE });
      audioCtxRef.current = ctx;
      await ctx.resume();

      const blob    = new Blob([WORKLET_CODE], { type: "application/javascript" });
      const blobURL = URL.createObjectURL(blob);
      await ctx.audioWorklet.addModule(blobURL);
      URL.revokeObjectURL(blobURL);

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = async () => {
        addLog("✅ VERA is online");
        setStatus("live");
        setVeraState("listening");

        const source      = ctx.createMediaStreamSource(audioStream);
        const workletNode = new AudioWorkletNode(ctx, "pcm-capture");
        workletNodeRef.current = workletNode;

        workletNode.port.onmessage = (e) => {
          const f32 = e.data;
          const i16 = new Int16Array(f32.length);
          for (let i = 0; i < f32.length; i++)
            i16[i] = Math.max(-32768, Math.min(32767, f32[i] * 32768));
          pcmBufferRef.current.push(i16);
        };

        source.connect(workletNode);
        flushTimerRef.current = setInterval(flushAudio, FLUSH_INTERVAL_MS);
        videoTimerRef.current = setInterval(sendVideo,  VIDEO_INTERVAL_MS);
        resetProactiveTimer();
        addLog("🎙️ Listening — speak to VERA");
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          switch (msg.type) {
            case "audio":
              handleAudioResponse(msg.data);
              setVeraState("speaking");
              resetProactiveTimer();
              break;
            case "turn_complete":
              setSessionTurns(t => t + 1);
              setVeraState("listening");
              resetProactiveTimer();
              addLog("✅ Listening...");
              break;
            case "interrupted":
              audioQueueRef.current = [];
              isPlayingRef.current  = false;
              setVeraState("listening");
              addLog("⚡ Interrupted");
              break;
            case "thinking":
              setVeraState("thinking");
              break;
            case "memory_saved": {
              const icons = { semantic:"👤", episodic:"📖", preference:"⚙️", events:"📅" };
              const icon  = icons[msg.category] || "🧠";
              addLog(`${icon} Remembered [${msg.category}]`);
              setMemories(prev => {
                const updated = [...prev, { category: msg.category, text: msg.text, icon, ts: new Date().toLocaleTimeString() }];
                return updated.slice(-30);
              });
              // Advance onboarding step when semantic memories are saved during onboarding
              if (msg.category === "semantic") {
                setOnboardStep(prev => Math.min(prev + 1, ONBOARD_STEPS.length - 1));
              }
              break;
            }
            case "error":
              addLog(`⚠️ ${msg.message}`);
              setVeraState("idle");
              break;
            case "pong":
              break;
            default:
              break;
          }
        } catch (e) { console.error("Parse error:", e); }
      };

      ws.onclose = (e) => {
        if (isStoppingRef.current) return;
        if (!mountedRef.current) return;
        addLog(`🔌 Connection lost — reconnecting in 3s...`);
        setStatus("idle");
        setVeraState("idle");
        setTimeout(() => {
          if (!isStoppingRef.current && mountedRef.current) {
            addLog("🔄 Auto-reconnecting...");
            startVERA();
          }
        }, 3000);
      };

      ws.onerror = () => addLog("⚠️ Connection error");

    } catch (err) {
      addLog(`❌ ${err.message}`);
      setStatus("error");
    }
  }, [cleanup, addLog, flushAudio, sendVideo, handleAudioResponse, resetProactiveTimer]);

  const stopVERA = useCallback(() => {
    isStoppingRef.current = true;
    clearTimeout(recapTimerRef.current);
    if (sessionTurns >= 3) {
      sendTextToVera("[SESSION_RECAP] Before we end, briefly summarize in 2 sentences what we discussed this session. Save it as an episodic memory.");
      recapTimerRef.current = setTimeout(() => {
        if (!mountedRef.current) return;
        cleanup();
        setStatus("idle");
        addLog("⏹️ VERA offline");
      }, 3000);
    } else {
      cleanup();
      setStatus("idle");
      addLog("⏹️ VERA offline");
    }
  }, [cleanup, addLog, sendTextToVera, sessionTurns]);

  useEffect(() => {
    mountedRef.current = true;

    // Fetch stored memory count from Firestore on mount
    // This tells us if VERA has met the host before (skip onboarding if yes)
    fetch(`${API_URL}/memories`)
      .then(r => r.json())
      .then(data => {
        if (!mountedRef.current) return;
        const total = Object.values(data).reduce((s, arr) => s + (Array.isArray(arr) ? arr.length : 0), 0);
        setStoredMemoryCount(total);
      })
      .catch(() => setStoredMemoryCount(0));

    return () => { mountedRef.current = false; cleanup(); };
  }, [cleanup]);

  // ── Restore video stream when returning from memory page ─────────────────────
  useEffect(() => {
    if (page === "main" && videoStreamRef.current) {
      const videoEl = document.getElementById("vera-video");
      if (videoEl && !videoEl.srcObject) {
        videoEl.srcObject = videoStreamRef.current;
        addLog("📷 Camera restored");
      }
    }
  }, [page, addLog]);

  // ── Camera watchdog — auto-restarts camera if stream dies ────────────────────
  const restartCamera = useCallback(async () => {
    if (!isStoppingRef.current && status === "live") {
      try {
        addLog("📷 Restarting camera...");
        // Stop old stream
        if (videoStreamRef.current) {
          videoStreamRef.current.getTracks().forEach(t => t.stop());
          videoStreamRef.current = null;
        }
        const videoStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: 320, height: 240 },
          audio: false,
        });
        videoStreamRef.current = videoStream;
        const videoEl = document.getElementById("vera-video");
        if (videoEl) videoEl.srcObject = videoStream;
        addLog("📷 Camera online");
      } catch (e) {
        addLog("📷 Camera retry failed — will try again");
      }
    }
  }, [status, addLog]);

  useEffect(() => {
    if (status !== "live") return;
    // Check camera health every 5 seconds
    cameraWatchdogRef.current = setInterval(() => {
      const videoEl = document.getElementById("vera-video");
      const stream  = videoStreamRef.current;
      const isDead  =
        !stream ||
        stream.getTracks().every(t => t.readyState === "ended") ||
        (videoEl && !videoEl.srcObject);
      if (isDead) restartCamera();
    }, 5000);
    return () => clearInterval(cameraWatchdogRef.current);
  }, [status, restartCamera]);

  // ── Onboarding detection ─────────────────────────────────────────────────────
  // Only triggers if storedMemoryCount is 0 (truly first session ever)
  // storedMemoryCount is fetched from Firestore on mount — not in-session memories
  useEffect(() => {
    if (status === "live" && storedMemoryCount === 0) {
      const timer = setTimeout(() => {
        setIsOnboarding(true);
        setOnboardStep(0);
        sendTextToVera(`[ONBOARDING] This appears to be your first conversation. Warmly introduce yourself as VERA and ask the user: "${ONBOARD_STEPS[0]}" — wait for their answer before asking the next question.`);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [status, storedMemoryCount, sendTextToVera]);

  const isLive       = status === "live";
  const isConnecting = status === "connecting";

  // ── ORB animation based on state ─────────────────────────────────────────────
  const orbAnimation = {
    idle:      "pulse-ring 3s ease-in-out infinite",
    listening: "pulse-ring 2s ease-in-out infinite",
    thinking:  "orb-think 1.5s ease-in-out infinite",
    speaking:  "orb-speak 0.8s ease-in-out infinite",
  }[veraState] || "pulse-ring 3s ease-in-out infinite";

  const orbColor = {
    idle:      "radial-gradient(circle at 35% 35%, #1a3a5c, #050a1a)",
    listening: "radial-gradient(circle at 35% 35%, #0a2a4a, #051020)",
    thinking:  "radial-gradient(circle at 35% 35%, #3a2000, #1a1000)",
    speaking:  "radial-gradient(circle at 35% 35%, #003a20, #001510)",
  }[veraState] || "radial-gradient(circle at 35% 35%, #1a3a5c, #050a1a)";

  const orbGlow = {
    idle:      "0 0 30px 6px #00d4ff33, 0 0 80px 15px #0066ff1a",
    listening: "0 0 40px 8px #00d4ff55, 0 0 100px 20px #0066ff2a",
    thinking:  "0 0 40px 8px #ff990055, 0 0 100px 20px #ff66002a",
    speaking:  "0 0 50px 10px #00ff9966, 0 0 120px 25px #00cc773a",
  }[veraState] || "0 0 30px 6px #00d4ff33";

  const stateLabel = {
    idle: "OFFLINE", listening: "LISTENING", thinking: "THINKING...", speaking: "SPEAKING"
  }[veraState] || "OFFLINE";

  const stateColor = {
    idle: "#444", listening: "#00d4ff", thinking: "#ff9900", speaking: "#00ff88"
  }[veraState] || "#444";

  // Memory categories display
  const memByCategory = {
    semantic:   memories.filter(m => m.category === "semantic"),
    preference: memories.filter(m => m.category === "preference"),
    events:     memories.filter(m => m.category === "events"),
    episodic:   memories.filter(m => m.category === "episodic"),
  };

  // Page routing
  if (page === "memory") {
    return <MemoryDashboard onBack={() => setPage("main")} />;
  }

  return (
    <div style={{
      minHeight: "100vh",
      background: "#030308",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "'Syne', sans-serif",
      padding: "20px",
      position: "relative",
      overflow: "hidden",
    }}>

      {/* Animated grid background */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none",
        backgroundImage: "linear-gradient(#0a1520 1px, transparent 1px), linear-gradient(90deg, #0a1520 1px, transparent 1px)",
        backgroundSize: "40px 40px",
        animation: "grid-drift 20s linear infinite",
        opacity: 0.4,
      }} />

      {/* Ambient glow top */}
      <div style={{
        position: "fixed", top: -200, left: "50%", transform: "translateX(-50%)",
        width: 600, height: 400,
        background: "radial-gradient(ellipse, #0066ff18 0%, transparent 70%)",
        pointerEvents: "none",
      }} />

      {/* Main card */}
      <div style={{
        width: "100%",
        maxWidth: 460,
        background: "rgba(8, 10, 20, 0.85)",
        border: "1px solid #0a2040",
        borderRadius: 24,
        padding: "40px 32px",
        backdropFilter: "blur(20px)",
        boxShadow: "0 0 0 1px #ffffff08, 0 40px 80px #000000aa",
        position: "relative",
        animation: "fadeIn 0.5s ease",
        zIndex: 1,
      }}>

        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 36 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 6 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: stateColor, boxShadow: `0 0 8px ${stateColor}` }} />
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: stateColor, letterSpacing: 4, fontWeight: 500 }}>
              {stateLabel}
            </span>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: stateColor, boxShadow: `0 0 8px ${stateColor}` }} />
          </div>
          <h1 style={{ margin: 0, fontSize: 56, fontWeight: 800, letterSpacing: 8, color: "#fff", lineHeight: 1 }}>VERA</h1>
          <p style={{ margin: "8px 0 0", fontSize: 11, color: "#3a5a7a", letterSpacing: 3, fontFamily: "'JetBrains Mono', monospace" }}>
            VOICE · ENHANCED · REALITY · ASSISTANT
          </p>
        </div>

        {/* ORB */}
        <div style={{ display: "flex", justifyContent: "center", marginBottom: 32 }}>
          <div style={{ position: "relative", width: 140, height: 140 }}>
            {/* Outer ring */}
            <div style={{
              position: "absolute", inset: -12,
              borderRadius: "50%",
              border: `1px solid ${stateColor}22`,
              animation: isLive ? orbAnimation : "none",
            }} />
            {/* Mid ring */}
            <div style={{
              position: "absolute", inset: -4,
              borderRadius: "50%",
              border: `1px solid ${stateColor}33`,
            }} />
            {/* Core orb */}
            <div style={{
              width: 140, height: 140,
              borderRadius: "50%",
              background: orbColor,
              boxShadow: isLive ? orbGlow : "0 0 20px 4px #00d4ff11",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: `1px solid ${stateColor}44`,
              transition: "background 0.5s ease, box-shadow 0.5s ease",
            }}>
              {/* Inner content based on state */}
              {veraState === "thinking" && <ThinkingDots />}
              {veraState === "speaking" && <WaveVisualizer active={true} color="#00ff88" />}
              {veraState === "listening" && <WaveVisualizer active={true} color="#00d4ff" />}
              {veraState === "idle" && (
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#1a4a6a", letterSpacing: 2 }}>VERA</span>
              )}
            </div>
          </div>
        </div>

        {/* Session turn counter */}
        {isLive && (
          <div style={{ textAlign: "center", marginBottom: 20 }}>
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#1a4a6a", letterSpacing: 2 }}>
              SESSION · {sessionTurns} TURNS · {memories.length} MEMORIES
            </span>
          </div>
        )}

        {/* Onboarding indicator */}
        {isOnboarding && isLive && (
          <div style={{
            background: "rgba(0, 100, 200, 0.1)",
            border: "1px solid #0044aa55",
            borderRadius: 12,
            padding: "10px 14px",
            marginBottom: 16,
            textAlign: "center",
            animation: "fadeInUp 0.3s ease",
          }}>
            <div style={{ color: "#4488ff", fontSize: 11, letterSpacing: 2, fontFamily: "'JetBrains Mono', monospace" }}>
              FIRST CONTACT · VERA IS LEARNING ABOUT YOU
            </div>
            <div style={{ display: "flex", justifyContent: "center", gap: 6, marginTop: 8 }}>
              {ONBOARD_STEPS.map((_, i) => (
                <div key={i} style={{
                  width: 20, height: 3, borderRadius: 2,
                  background: i <= onboardStep ? "#4488ff" : "#1a2a3a",
                  transition: "background 0.3s ease",
                }} />
              ))}
            </div>
          </div>
        )}

        {/* Video feed */}
        <video id="vera-video" autoPlay muted playsInline style={{
          width: "100%", borderRadius: 14,
          marginBottom: 20,
          border: "1px solid #0a2040",
          display: isLive ? "block" : "none",
          boxShadow: "0 0 20px #00d4ff0a",
        }} />
        <canvas ref={videoCanvasRef} style={{ display: "none" }} />

        {/* CTA Button */}
        {!isLive && !isConnecting ? (
          <button className="vera-btn" onClick={startVERA} style={{
            width: "100%", padding: "16px 0",
            background: "linear-gradient(135deg, #0044cc, #0066ff)",
            color: "#fff", border: "none", borderRadius: 14,
            fontSize: 14, fontWeight: 700, cursor: "pointer",
            letterSpacing: 3, fontFamily: "'Syne', sans-serif",
            marginBottom: 16,
            boxShadow: "0 4px 30px #0066ff44",
          }}>▶ ACTIVATE VERA</button>
        ) : isConnecting ? (
          <button disabled style={{
            width: "100%", padding: "16px 0",
            background: "#0a1a2a", color: "#1a4a6a",
            border: "1px solid #0a2040", borderRadius: 14,
            fontSize: 14, fontWeight: 700, letterSpacing: 3,
            fontFamily: "'Syne', sans-serif", marginBottom: 16,
          }}>INITIALIZING...</button>
        ) : (
          <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
            <button className="vera-btn" onClick={stopVERA} style={{
              flex: 1, padding: "14px 0",
              background: "rgba(180, 30, 30, 0.15)",
              color: "#ff4444", border: "1px solid #ff444433",
              borderRadius: 14, fontSize: 13, fontWeight: 700,
              cursor: "pointer", letterSpacing: 2,
              fontFamily: "'Syne', sans-serif",
            }}>⏹ DEACTIVATE</button>
            <button className="vera-btn" onClick={() => setPage("memory")} style={{
              flex: 1, padding: "14px 0",
              background: memPanel ? "rgba(0, 100, 255, 0.2)" : "rgba(0, 100, 255, 0.08)",
              color: "#4488ff", border: "1px solid #4488ff33",
              borderRadius: 14, fontSize: 13, fontWeight: 700,
              cursor: "pointer", letterSpacing: 2,
              fontFamily: "'Syne', sans-serif",
            }}>🧠 MEMORY {memories.length > 0 ? `(${memories.length})` : ""}</button>
          </div>
        )}

        {/* Memory Panel */}
        {memPanel && (
          <div style={{
            background: "rgba(0, 10, 25, 0.9)",
            border: "1px solid #0a2a4a",
            borderRadius: 14,
            padding: "16px",
            marginBottom: 16,
            maxHeight: 280,
            overflowY: "auto",
            animation: "fadeInUp 0.3s ease",
          }}>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#1a5a8a", letterSpacing: 3, marginBottom: 12 }}>
              VERA'S MEMORY BANK
            </div>
            {memories.length === 0 ? (
              <div style={{ color: "#1a3a5a", fontSize: 12, textAlign: "center", padding: "20px 0" }}>
                No memories yet — start talking to VERA
              </div>
            ) : (
              Object.entries(memByCategory).map(([cat, items]) => items.length > 0 && (
                <div key={cat} style={{ marginBottom: 14 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                    <div style={{ width: 2, height: 12, background: CAT_COLORS[cat], borderRadius: 1 }} />
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: CAT_COLORS[cat], letterSpacing: 2, textTransform: "uppercase" }}>
                      {cat}
                    </span>
                  </div>
                  {items.slice(-3).map((m, i) => (
                    <div key={i} className="mem-entry" style={{
                      padding: "6px 10px",
                      background: `${CAT_COLORS[cat]}11`,
                      border: `1px solid ${CAT_COLORS[cat]}22`,
                      borderRadius: 8,
                      marginBottom: 4,
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 8,
                    }}>
                      <span style={{ color: "#7a9ab0", fontSize: 12, lineHeight: 1.4, flex: 1 }}>{m.text}</span>
                      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#1a3a5a", flexShrink: 0 }}>{m.ts}</span>
                    </div>
                  ))}
                </div>
              ))
            )}
          </div>
        )}

        {/* Activity Log */}
        <div style={{
          background: "rgba(0, 5, 15, 0.8)",
          border: "1px solid #0a1a2a",
          borderRadius: 12,
          padding: "12px 14px",
          minHeight: 100,
          maxHeight: 160,
          overflowY: "auto",
        }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#0a2a3a", letterSpacing: 3, marginBottom: 8 }}>
            ACTIVITY LOG
          </div>
          {log.length === 0 ? (
            <span style={{ color: "#0a2a3a", fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }}>
              Awaiting activation...
            </span>
          ) : (
            log.map((l, i) => (
              <div key={i} className="log-entry" style={{
                color: i === log.length - 1 ? "#5a8ab0" : "#2a4a6a",
                fontSize: 12,
                lineHeight: "1.8",
                fontFamily: "'JetBrains Mono', monospace",
                transition: "color 0.3s ease",
              }}>{l}</div>
            ))
          )}
        </div>

        {/* Footer */}
        <div style={{ textAlign: "center", marginTop: 16 }}>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#0a1a2a", letterSpacing: 2 }}>
            POWERED BY GEMINI LIVE · GOOGLE AI
          </span>
        </div>
      </div>
    </div>
  );
}

const CAT_COLORS = {
  semantic:   "#4488ff",
  preference: "#ff9900",
  events:     "#aa44ff",
  episodic:   "#00cc77",
};