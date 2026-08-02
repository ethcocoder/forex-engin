import { useEffect, useState } from "react"
import { NavLink, Outlet, useNavigate } from "react-router-dom"
import { LayoutDashboard, Activity, ArrowLeftRight, FileText, Settings2, Play, Square } from "lucide-react"
import { useLatest } from "../hooks"
import "../styles/app.css"

const NAV = [
  { to: "/", label: "Dashboard", end: true, icon: LayoutDashboard },
  { to: "/signals", label: "Signals", icon: Activity },
  { to: "/trades", label: "Trades", icon: ArrowLeftRight },
  { to: "/reports", label: "Reports", icon: FileText },
  { to: "/settings", label: "Settings", icon: Settings2 }
]

function fmt(value: number | undefined, digits = 2): string {
  return value == null ? "—" : value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

function pnlClass(v: number | undefined): string {
  if (v == null) return ""
  return v > 0 ? "pos" : v < 0 ? "neg" : ""
}

export default function Layout(): React.JSX.Element {
  const account = useLatest("account")
  const positions = useLatest("positions")
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
  const balance = typeof a?.cash === "number" ? (a.cash as number) : undefined
  const equity = typeof a?.equity === "number" ? (a.equity as number) : undefined
  const pnl = typeof a?.daily_pnl === "number" ? (a.daily_pnl as number) : undefined

  const posData = (positions?.data ?? {}) as Record<string, { size?: number }>
  const posCount = Object.keys(posData).length

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

        <div className="sidebar-sep" />

        <nav className="nav">
          {NAV.slice(0, 4).map((n) => {
            const Icon = n.icon
            return (
              <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => (isActive ? "active" : "")}>
                <Icon />
                {n.label}
              </NavLink>
            )
          })}
        </nav>

        <div className="sidebar-sep" />

        <nav className="nav">
          {NAV.slice(4).map((n) => {
            const Icon = n.icon
            return (
              <NavLink key={n.to} to={n.to} className={({ isActive }) => (isActive ? "active" : "")}>
                <Icon />
                {n.label}
              </NavLink>
            )
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="version">v0.1.0</div>
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          <div className="stat">
            <span className="lbl">Balance</span>
            <span className="val num">${fmt(balance)}</span>
          </div>
          <div className="stat sep">
            <span className="lbl">Equity</span>
            <span className="val num">${fmt(equity)}</span>
          </div>
          <div className="stat sep">
            <span className="lbl">Daily PnL</span>
            <span className={`val num ${pnlClass(pnl)}`}>{pnl == null ? "—" : `${pnl >= 0 ? "+" : ""}$${fmt(pnl)}`}</span>
          </div>
          <div className="stat sep">
            <span className="lbl">Positions</span>
            <span className="val num">{posCount === 0 ? "—" : String(posCount)}</span>
          </div>

          <div className="spacer" />

          <div className="actions">
            <button className="icon-btn primary" title="Run simulation (Ctrl+R)" disabled={busy || ["running", "starting"].includes(simState)} onClick={() => void onRun()}>
              <Play />
            </button>
            <button className="icon-btn danger" title="Stop simulation (Ctrl+.)" disabled={busy || !["running", "starting", "stopping"].includes(simState)} onClick={() => void onStop()}>
              <Square />
            </button>
          </div>
        </div>

        <div className="content">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
