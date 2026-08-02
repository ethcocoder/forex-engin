import { useEffect, useState } from "react"
import { NavLink, Outlet, useNavigate } from "react-router-dom"
import { useLatest } from "../hooks"
import "../styles/app.css"

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/signals", label: "Signals" },
  { to: "/trades", label: "Trades" },
  { to: "/reports", label: "Reports" },
  { to: "/settings", label: "Settings" }
]

function fmt(value: number | undefined, digits = 2): string {
  return value == null ? "—" : value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

function pnlClass(v: number | undefined): string {
  if (v == null) return ""
  return v > 0 ? "pos" : v < 0 ? "neg" : ""
}

const SIM_PILL: Record<string, { label: string; cls: string }> = {
  idle: { label: "IDLE", cls: "" },
  starting: { label: "STARTING", cls: "busy" },
  loading: { label: "LOADING", cls: "busy" },
  running: { label: "RUNNING", cls: "busy" },
  stopping: { label: "STOPPING", cls: "busy" },
  done: { label: "DONE", cls: "ok" },
  error: { label: "ERROR", cls: "err" }
}

export default function Layout(): React.JSX.Element {
  const account = useLatest("account")
  const positions = useLatest("positions")
  const signal = useLatest("signal")
  const engineStatus = useLatest("engine.status")
  const simStatus = useLatest("sim.status") ?? useLatest("status")
  const [busy, setBusy] = useState(false)
  const [simState, setSimState] = useState<string>("idle")
  const navigate = useNavigate()

  useEffect(() => {
    void window.api.sim.status().then((s) => setSimState(s.status))
  }, [])

  useEffect(() => {
    if (simStatus) setSimState(String(simStatus.data.status ?? ""))
  }, [simStatus])

  const a = account?.data as Record<string, unknown> | undefined
  const connected = engineStatus?.data?.state === "connected"
  const balance = typeof a?.cash === "number" ? (a.cash as number) : undefined
  const equity = typeof a?.equity === "number" ? (a.equity as number) : undefined
  const pnl = typeof a?.daily_pnl === "number" ? (a.daily_pnl as number) : undefined

  const posData = (positions?.data ?? {}) as Record<string, { size?: number }>
  const posCount = Object.keys(posData).length
  const notional = Object.values(posData).reduce((sum, p) => sum + Math.abs(Number(p.size ?? 0)), 0)
  const regime = Number((signal?.data as Record<string, unknown> | undefined)?.regime ?? NaN)
  const pill = SIM_PILL[simState] ?? SIM_PILL.idle!

  const onRun = async (): Promise<void> => {
    if (busy || ["running", "starting"].includes(simState)) return
    setBusy(true)
    try {
      const res = await window.api.sim.start()
      if (res.ok) setSimState("starting")
      else setSimState("error")
    } finally {
      setBusy(false)
    }
  }

  const onStop = async (): Promise<void> => {
    if (busy || !["running", "starting", "stopping"].includes(simState)) return
    setBusy(true)
    try {
      await window.api.sim.stop()
      setSimState("stopping")
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (!(e.ctrlKey || e.metaKey)) return
      const k = e.key.toLowerCase()
      if (k === "r") {
        e.preventDefault()
        void onRun()
      } else if (k === ".") {
        e.preventDefault()
        void onStop()
      } else if (k >= "1" && k <= "5") {
        const item = NAV[Number(k) - 1]
        if (item) {
          e.preventDefault()
          navigate(item.to)
        }
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy, simState])

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand wordmark">
          FOREX<span className="b">DESK</span>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => (isActive ? "active" : "")}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="muted" style={{ padding: "10px", fontSize: 11 }}>
          <span className={`status-dot ${connected ? "ok" : "bad"}`} />
          {connected ? "Engine connected" : "Engine offline"}
        </div>
      </aside>
      <div className="main">
        <div className="topbar">
          <div className="stat">
            <span className="lbl">Balance</span>
            <span className="val num">${fmt(balance)}</span>
          </div>
          <div className="stat">
            <span className="lbl">Equity</span>
            <span className="val num">${fmt(equity)}</span>
          </div>
          <div className="stat">
            <span className="lbl">Daily PnL</span>
            <span className={`val num ${pnlClass(pnl)}`}>{pnl == null ? "—" : `${pnl >= 0 ? "+" : ""}$${fmt(pnl)}`}</span>
          </div>
          <div className="stat">
            <span className="lbl">Open Positions</span>
            <span className="val num">{posCount === 0 ? "—" : `${posCount} · ${fmt(notional, 0)}`}</span>
          </div>
          <div className="stat">
            <span className="lbl">Regime</span>
            <span className="val num">{Number.isNaN(regime) ? "—" : String(regime)}</span>
          </div>
          <div className="spacer" />
          <div className={`pill ${pill.cls}`}>{pill.label}</div>
          <div className="badge">SIMULATION</div>
          {connected && (
            <>
              <button className="btn" disabled={busy || ["running", "starting"].includes(simState)} onClick={() => void onRun()}>
                Run
              </button>
              <button className="btn danger" disabled={busy || !["running", "starting", "stopping"].includes(simState)} onClick={() => void onStop()}>
                Stop
              </button>
            </>
          )}
        </div>
        <div className="content">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
