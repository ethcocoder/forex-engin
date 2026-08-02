import { useEffect, useState } from "react"
import { ArrowUp, ArrowDown } from "lucide-react"
import { useEvents, useLatest } from "../hooks"
import { dirClass, fmtNum, fmtPct, fmtTs } from "../format"
import EquityChart, { type EquityPoint } from "../components/EquityChart"

interface Metric {
  label: string
  value: string
  cls?: string
}

function useMetrics(lastEquity?: number): Metric[] {
  const account = useLatest("account")
  const tearSheet = useLatest("reports.tear-sheet")
  const [ts, setTs] = useState<Record<string, number> | null>(null)

  useEffect(() => {
    void window.api.reports.tearSheet().then((r) => setTs(r.metrics)).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (tearSheet?.data?.metrics) setTs(tearSheet.data.metrics as Record<string, number>)
  }, [tearSheet])

  const a = account?.data as Record<string, unknown> | undefined
  const pnl = typeof a?.daily_pnl === "number" ? (a.daily_pnl as number) : undefined
  const win = typeof a?.win_rate === "number" ? (a.win_rate as number) : undefined
  const liveEquity = typeof a?.equity === "number" ? (a.equity as number) : undefined

  return [
    { label: "Equity", value: liveEquity != null ? `$${fmtNum(liveEquity)}` : lastEquity != null ? `$${fmtNum(lastEquity)}` : "—" },
    {
      label: "P&L Today",
      value: pnl == null ? "—" : `${pnl >= 0 ? "+" : ""}$${fmtNum(pnl)}`,
      cls: pnl == null ? "" : pnl > 0 ? "pos" : "neg"
    },
    { label: "Sharpe", value: ts?.sharpe == null ? "—" : fmtNum(ts.sharpe, 2) },
    { label: "Win Rate", value: win == null ? "—" : fmtPct(win * 100) }
  ]
}

function useEquityCurve(): EquityPoint[] {
  const live = useEvents("equity")
  const [curve, setCurve] = useState<EquityPoint[]>([])
  const [fetched, setFetched] = useState(false)

  useEffect(() => {
    void window.api.engine
      .equity()
      .then((r) => {
        setCurve(r.equity.map((e) => ({ ts: e.ts, equity: e.equity })))
        setFetched(true)
      })
      .catch(() => setFetched(true))
  }, [])

  useEffect(() => {
    if (!fetched) return
    setCurve((prev) => {
      const next = [...prev]
      for (const e of [...live].reverse()) {
        const pt: EquityPoint = { ts: e.ts, equity: Number(e.data.equity) }
        const last = next[next.length - 1]
        if (last && last.ts === pt.ts) next[next.length - 1] = pt
        else next.push(pt)
      }
      return next.length > 3000 ? next.slice(-3000) : next
    })
  }, [live, fetched])

  return curve
}

function Ticker(): React.JSX.Element {
  const signal = useLatest("signal")
  if (!signal) return <div className="empty">No signal yet — start a simulation.</div>
  const d = signal.data as Record<string, unknown>
  const conf = Number(d.confidence ?? 0)
  const dir = Number(d.direction ?? 0)
  const cls = conf >= 0.6 ? "" : "warn"
  return (
    <div className="ticker">
      <span className={`tick-dir num ${dirClass(dir)}`}>
        {dir > 0 ? <ArrowUp /> : dir < 0 ? <ArrowDown /> : null}
      </span>
      <div className="tick-main">
        <span className="tick-mag num">{fmtNum(d.magnitude, 3)}</span>
        <span className="tick-sub muted">
          confidence {fmtPct(conf * 100)} · uncertainty {fmtNum(d.uncertainty, 3)} · regime {String(d.regime ?? "—")}
        </span>
      </div>
      <span className={`tick-conf num ${cls}`}>{fmtPct(conf * 100, 0)}</span>
    </div>
  )
}

export default function Dashboard(): React.JSX.Element {
  const curve = useEquityCurve()
  const lastEquity = curve.length > 0 ? curve[curve.length - 1].equity : undefined
  const metrics = useMetrics(lastEquity)
  const signals = useEvents("signal")
  const trades = useEvents("trade")

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid cols-4">
        {metrics.map((m) => (
          <div className="card metric" key={m.label}>
            <div className="lbl">{m.label}</div>
            <div className={`val num ${m.cls ?? ""}`}>{m.value}</div>
          </div>
        ))}
      </div>

      <div className="card">
        <div className="card-head">
          <h3>Equity Curve</h3>
        </div>
        <EquityChart points={curve} />
      </div>

      <div className="card">
        <div className="card-head">
          <h3>Live Signal</h3>
        </div>
        <Ticker />
      </div>

      <div className="grid cols-2">
        <div className="card">
          <div className="card-head">
            <h3>Recent Trades</h3>
          </div>
          {trades.length === 0 ? (
            <div className="empty">No closed trades yet.</div>
          ) : (
            <table className="data">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Dir</th>
                  <th>Pnl</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {trades.slice(0, 8).map((t, i) => {
                  const d = t.data
                  const pnl = Number(d.pnl ?? 0)
                  const dir = Number(d.direction ?? 0)
                  return (
                    <tr key={i}>
                      <td className="num">{fmtTs(t.ts)}</td>
                      <td className={`num ${dirClass(d.direction)}`}>
                        {dir > 0 ? <ArrowUp size={14} /> : dir < 0 ? <ArrowDown size={14} /> : "—"}
                      </td>
                      <td className={`num ${pnl > 0 ? "pos" : pnl < 0 ? "neg" : ""}`}>{pnl >= 0 ? "+" : ""}{fmtNum(pnl)}</td>
                      <td>{String(d.exit_reason ?? "—")}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>

        <div className="card">
          <div className="card-head">
            <h3>Recent Signals</h3>
          </div>
          {signals.length === 0 ? (
            <div className="empty">No signals yet — start a simulation.</div>
          ) : (
            <table className="data">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Dir</th>
                  <th>Confidence</th>
                  <th>Regime</th>
                </tr>
              </thead>
              <tbody>
                {signals.slice(0, 8).map((s, i) => {
                  const d = s.data
                  const dir = Number(d.direction ?? 0)
                  return (
                    <tr key={i}>
                      <td className="num">{fmtTs(s.ts)}</td>
                      <td className={`num ${dirClass(d.direction)}`}>
                        {dir > 0 ? <ArrowUp size={14} /> : dir < 0 ? <ArrowDown size={14} /> : "—"}
                      </td>
                      <td className="num">{fmtNum(d.confidence, 3)}</td>
                      <td className="num">{String(d.regime ?? "—")}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
