// MemoryDashboard.jsx - VERA Memory Dashboard
// Supports all 5 memory categories: semantic, preference, events, episodic, speaker
import { useState, useEffect, useCallback } from "react";

const API = "https://vera-assistant-production.up.railway.app";

const CATS = {
  semantic:   { label: "Identity",    icon: "👤", color: "#00d4ff", desc: "Who you are" },
  preference: { label: "Preferences", icon: "⚙️", color: "#a78bfa", desc: "How you like things" },
  events:     { label: "Events",      icon: "📅", color: "#34d399", desc: "Meetings & reminders" },
  episodic:   { label: "Sessions",    icon: "📖", color: "#fb923c", desc: "Conversation history" },
  speaker:    { label: "People",      icon: "🧑", color: "#f472b6", desc: "People you've met" },
};

// Maps category name → the Firestore field that holds the text value
const FIELD = {
  semantic:   "fact",
  preference: "pattern",
  events:     "title",
  episodic:   "summary",
  speaker:    "profile",
};

const FONT_LINK = document.createElement("link");
FONT_LINK.rel = "stylesheet";
FONT_LINK.href = "https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap";
document.head.appendChild(FONT_LINK);

export default function MemoryDashboard({ onBack }) {
  const [memories, setMemories]     = useState({});
  const [logs, setLogs]             = useState([]);
  const [loading, setLoading]       = useState(true);
  const [logsLoading, setLogsLoading] = useState(false);
  const [search, setSearch]         = useState("");
  const [activeTab, setActiveTab]   = useState("all");
  const [activeView, setActiveView] = useState("memories"); // "memories" | "logs"
  const [deleting, setDeleting]     = useState(null);
  const [error, setError]           = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/memories`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setMemories(data);
    } catch (e) {
      console.error("Failed to load memories", e);
      setError("Could not reach VERA backend. Is Railway running?");
    }
    setLoading(false);
  }, []);

  const loadLogs = useCallback(async () => {
    setLogsLoading(true);
    try {
      const res = await fetch(`${API}/logs`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setLogs(data.logs || []);
    } catch (e) {
      console.error("Failed to load logs", e);
    }
    setLogsLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    if (activeView === "logs" && logs.length === 0) loadLogs();
  }, [activeView, logs.length, loadLogs]);

  const clearCategory = async (cat) => {
    if (!window.confirm(`Clear all ${CATS[cat].label} memories?`)) return;
    setDeleting(cat);
    try {
      await fetch(`${API}/memories/${cat}`, { method: "DELETE" });
    } catch (e) {
      console.error("Clear failed", e);
    }
    setDeleting(null);
    load();
  };

  // Flatten all memory entries from all categories
  const allEntries = Object.entries(memories).flatMap(([cat, items]) => {
    if (!CATS[cat] || !Array.isArray(items)) return [];
    return items.map(item => ({
      cat,
      text: item[FIELD[cat]] || item.fact || item.summary || item.pattern || item.title || item.profile || "",
      ts:   item.timestamp || "",
      item,
    })).filter(e => e.text);
  });

  const filtered = allEntries.filter(e => {
    const matchTab    = activeTab === "all" || e.cat === activeTab;
    const matchSearch = !search || e.text.toLowerCase().includes(search.toLowerCase());
    return matchTab && matchSearch;
  }).sort((a, b) => b.ts.localeCompare(a.ts));

  const totalCount = allEntries.length;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#030308",
      color: "#e2e8f0",
      fontFamily: "'Syne', sans-serif",
      padding: 24,
    }}>

      {/* ── Header ── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 32 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {onBack && (
            <button onClick={onBack} style={{
              background: "transparent", border: "1px solid #ffffff22",
              color: "#94a3b8", borderRadius: 8, padding: "8px 16px",
              cursor: "pointer", fontSize: 13, fontFamily: "inherit",
            }}>← Back</button>
          )}
          <div>
            <h1 style={{
              margin: 0, fontSize: 28, fontWeight: 800,
              background: "linear-gradient(135deg, #00d4ff, #a78bfa)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              VERA Memory
            </h1>
            <p style={{ margin: 0, fontSize: 13, color: "#64748b" }}>
              {totalCount} memories stored permanently
            </p>
          </div>
        </div>

        {/* View toggle + refresh */}
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => setActiveView("memories")} style={{
            background: activeView === "memories" ? "#00d4ff18" : "#ffffff08",
            border: `1px solid ${activeView === "memories" ? "#00d4ff44" : "#ffffff15"}`,
            color: activeView === "memories" ? "#00d4ff" : "#94a3b8",
            borderRadius: 8, padding: "8px 14px",
            cursor: "pointer", fontSize: 13, fontFamily: "inherit",
          }}>🧠 Memories</button>
          <button onClick={() => setActiveView("logs")} style={{
            background: activeView === "logs" ? "#a78bfa18" : "#ffffff08",
            border: `1px solid ${activeView === "logs" ? "#a78bfa44" : "#ffffff15"}`,
            color: activeView === "logs" ? "#a78bfa" : "#94a3b8",
            borderRadius: 8, padding: "8px 14px",
            cursor: "pointer", fontSize: 13, fontFamily: "inherit",
          }}>📋 Logs</button>
          <button onClick={activeView === "memories" ? load : loadLogs} style={{
            background: "#ffffff08", border: "1px solid #ffffff15",
            color: "#94a3b8", borderRadius: 8, padding: "8px 14px",
            cursor: "pointer", fontSize: 13, fontFamily: "inherit",
          }}>↻ Refresh</button>
        </div>
      </div>

      {/* ── Error Banner ── */}
      {error && (
        <div style={{
          background: "#ff444415", border: "1px solid #ff444433",
          borderRadius: 10, padding: "12px 16px", marginBottom: 24,
          color: "#ff6b6b", fontSize: 13,
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* ══════════════════════════════════════════ */}
      {/* MEMORIES VIEW                             */}
      {/* ══════════════════════════════════════════ */}
      {activeView === "memories" && (
        <>
          {/* Category Cards */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: 14, marginBottom: 28,
          }}>
            {Object.entries(CATS).map(([cat, meta]) => {
              const count = (memories[cat] || []).length;
              return (
                <div key={cat}
                  onClick={() => setActiveTab(activeTab === cat ? "all" : cat)}
                  style={{
                    background: activeTab === cat ? `${meta.color}18` : "#ffffff06",
                    border: `1px solid ${activeTab === cat ? meta.color + "55" : "#ffffff12"}`,
                    borderRadius: 12, padding: 18, cursor: "pointer",
                    transition: "all 0.2s",
                  }}>
                  <div style={{ fontSize: 26, marginBottom: 6 }}>{meta.icon}</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color: meta.color }}>{count}</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginTop: 2 }}>{meta.label}</div>
                  <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{meta.desc}</div>
                  {count > 0 && (
                    <button
                      onClick={e => { e.stopPropagation(); clearCategory(cat); }}
                      style={{
                        marginTop: 10, background: "#ff444408",
                        border: "1px solid #ff444433", color: "#ff6b6b",
                        borderRadius: 6, padding: "4px 10px",
                        cursor: "pointer", fontSize: 11, fontFamily: "inherit",
                        opacity: deleting === cat ? 0.5 : 1,
                      }}>
                      {deleting === cat ? "Clearing..." : "Clear"}
                    </button>
                  )}
                </div>
              );
            })}
          </div>

          {/* Search */}
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search memories..."
            style={{
              width: "100%", background: "#ffffff08",
              border: "1px solid #ffffff15", borderRadius: 10,
              padding: "12px 16px", color: "#e2e8f0",
              fontSize: 14, fontFamily: "inherit", outline: "none",
              boxSizing: "border-box", marginBottom: 16,
            }}
          />

          {/* Tab Filter */}
          <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
            {[["all", "🧠", "All"], ...Object.entries(CATS).map(([k, v]) => [k, v.icon, v.label])].map(([key, icon, label]) => (
              <button key={key} onClick={() => setActiveTab(key)} style={{
                background: activeTab === key ? "#00d4ff18" : "#ffffff06",
                border: `1px solid ${activeTab === key ? "#00d4ff44" : "#ffffff12"}`,
                color: activeTab === key ? "#00d4ff" : "#94a3b8",
                borderRadius: 20, padding: "6px 14px",
                cursor: "pointer", fontSize: 12, fontFamily: "inherit",
              }}>{icon} {label}</button>
            ))}
          </div>

          {/* Memory List */}
          {loading ? (
            <div style={{ textAlign: "center", padding: 60, color: "#64748b" }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>🧠</div>
              Loading memories from Firestore...
            </div>
          ) : filtered.length === 0 ? (
            <div style={{ textAlign: "center", padding: 60, color: "#64748b" }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>
                {search ? "🔍" : "💭"}
              </div>
              {search
                ? "No memories match your search."
                : totalCount === 0
                  ? "No memories yet — start talking to VERA!"
                  : "No memories in this category."}
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {filtered.map((entry, i) => {
                const meta = CATS[entry.cat];
                if (!meta) return null;
                return (
                  <div key={i} style={{
                    background: "#ffffff06",
                    border: "1px solid #ffffff0e",
                    borderLeft: `3px solid ${meta.color}`,
                    borderRadius: 10, padding: "14px 18px",
                    display: "flex", alignItems: "flex-start", gap: 14,
                  }}>
                    <span style={{ fontSize: 20, flexShrink: 0 }}>{meta.icon}</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        fontSize: 14, color: "#e2e8f0",
                        lineHeight: 1.6, wordBreak: "break-word",
                      }}>
                        {entry.text}
                      </div>
                      <div style={{ display: "flex", gap: 10, marginTop: 8, alignItems: "center", flexWrap: "wrap" }}>
                        <span style={{
                          fontSize: 10, color: meta.color,
                          background: `${meta.color}15`,
                          padding: "2px 8px", borderRadius: 10,
                          textTransform: "uppercase", letterSpacing: 1,
                          fontFamily: "'JetBrains Mono', monospace",
                        }}>{meta.label}</span>
                        {entry.ts && (
                          <span style={{
                            fontSize: 11, color: "#475569",
                            fontFamily: "'JetBrains Mono', monospace",
                          }}>{entry.ts}</span>
                        )}
                        {entry.cat === "events" && (
                          <span style={{
                            fontSize: 10,
                            color: entry.item.completed ? "#34d399" : "#fb923c",
                            background: entry.item.completed ? "#34d39915" : "#fb923c15",
                            padding: "2px 8px", borderRadius: 10,
                          }}>
                            {entry.item.completed ? "✓ Done" : "Pending"}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* ══════════════════════════════════════════ */}
      {/* LOGS VIEW                                 */}
      {/* ══════════════════════════════════════════ */}
      {activeView === "logs" && (
        <>
          {logsLoading ? (
            <div style={{ textAlign: "center", padding: 60, color: "#64748b" }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>📋</div>
              Loading conversation logs...
            </div>
          ) : logs.length === 0 ? (
            <div style={{ textAlign: "center", padding: 60, color: "#64748b" }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>📭</div>
              No conversation logs yet. Logs are saved when you deactivate VERA.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {[...logs].reverse().map((log, i) => (
                <div key={i} style={{
                  background: "#ffffff06", border: "1px solid #ffffff0e",
                  borderRadius: 12, padding: 20,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 14 }}>
                    <span style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 11, color: "#a78bfa", letterSpacing: 2,
                    }}>
                      📋 SESSION · {log.turns} TURNS
                    </span>
                    <span style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 11, color: "#475569",
                    }}>{log.timestamp}</span>
                  </div>
                  <div style={{
                    background: "#000000aa", borderRadius: 8,
                    padding: "12px 16px", maxHeight: 300, overflowY: "auto",
                  }}>
                    {(log.log || "").split("\n").map((line, j) => {
                      const isHost = line.startsWith("Host:");
                      const isVera = line.startsWith("VERA:");
                      return (
                        <div key={j} style={{
                          fontFamily: "'JetBrains Mono', monospace",
                          fontSize: 12,
                          color: isHost ? "#00d4ff" : isVera ? "#a78bfa" : "#475569",
                          lineHeight: 1.8,
                          padding: "1px 0",
                        }}>
                          {line}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Footer */}
      <div style={{ textAlign: "center", marginTop: 40, paddingTop: 24, borderTop: "1px solid #ffffff08" }}>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#1a2a3a", letterSpacing: 2 }}>
          VERA MEMORY SYSTEM · FIRESTORE · PERSISTENT
        </span>
      </div>
    </div>
  );
}