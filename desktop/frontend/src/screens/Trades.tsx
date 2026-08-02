import { useEffect, useMemo, useState } from "react"
import { ArrowUp, ArrowDown } from "lucide-react"
import { useEvents } from "../hooks"
import { dirClass, fmtHold, fmtNum, fmtTs } from "../format"
import type { TradeRow } from "../../electron/audit"

type Filter = "all" | string

function toRow(t: unknown, ts?: number): TradeRow {
  const d = t as Record<string, unknown>
  const num = (k: string): number => {
    const v = Number(d[k] ?? 0)
    return Number.isFinite(v) ? v : 0
  }
  return {
    id: Number(d.id ?? 0),
    ts: typeof ts === "number" ? ts : num("ts") || Date.now() / 1000,
    pair: String(d.pair ?? ""),
    direction: num("direction"),
    size: num("size"),
    entry_price: num("entry_price"),
    exit_price: num("exit_price"),
    pnl: num("pnl"),
    slippage_pips: num("slippage_pips"),
    hold_steps: Math.round(num("hold_steps")),
    exit_reason: String(d.exit_reason ?? "")
  }
}

const key = (t: TradeRow): string => `${t.ts}|${t.pair}|${t.pnl}`

export default function Trades(): React.JSX.Element {
  const live = useEvents("trade")
  const orders = useEvents("order")
  const [audit, setAudit] = useState<TradeRow[]>([])
  const [reason, setReason] = useState<Filter>("all")
  const [pair, setPair] = useState<Filter>("all")

  useEffect(() => {
    void window.api.audit.trades(500).then((rows) => setAudit((rows as TradeRow[]).map((r) => toRow(r, r.ts)))).catch(() => undefined)
  }, [])

  const rows = useMemo(() => {
    const seen = new Set<string>()
    const out: TradeRow[] = []
    for (const l of live) {
      const r = toRow(l.data, l.ts)
      const k = key(r)
      if (!seen.has(k)) {
        seen.add(k)
        out.push(r)
      }
    }
    for (const a of audit) {
      const k = key(a)
      if (!seen.has(k)) {
        seen.add(k)
        out.push(a)
      }
    }
    return out
  }, [live, audit])

  const reasons = useMemo(() => ["all", ...new Set(rows.map((r) => r.exit_reason).filter(Boolean))].sort(), [rows])
  const pairs = useMemo(() => ["all", ...new Set(rows.map((r) => r.pair).filter(Boolean))].sort(), [rows])

  const filtered = useMemo(
    () => rows.filter((r) => (reason === "all" || r.exit_reason === reason) && (pair === "all" || r.pair === pair)),
    [rows, reason, pair]
  )
  const totalPnl = filtered.reduce((s, r) => s + r.pnl, 0)
  const maxAbs = useMemo(() => Math.max(...filtered.map((r) => Math.abs(r.pnl)), 1), [filtered])

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="card">
        <div className="card-head">
          <h3>Trades</h3>
          <div className="filters">
            <select className="input" value={reason} onChange={(e) => setReason(e.target.value)}>
              {reasons.map((r) => (
                <option key={r} value={r}>{r === "all" ? "All reasons" : r}</option>
              ))}
            </select>
            <select className="input" value={pair} onChange={(e) => setPair(e.target.value)}>
              {pairs.map((p) => (
                <option key={p} value={p}>{p === "all" ? "All pairs" : p}</option>
              ))}
            </select>
            <span className="muted num">
              {filtered.length} trades · Σ <span className={totalPnl > 0 ? "pos" : totalPnl < 0 ? "neg" : ""}>{totalPnl >= 0 ? "+" : ""}{fmtNum(totalPnl)}</span>
            </span>
          </div>
        </div>

        {rows.length === 0 ? (
          <div className="empty">No trades in the audit log yet.</div>
        ) : (
          <div className="scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Pair</th>
                  <th>Dir</th>
                  <th>Size</th>
                  <th>Entry</th>
                  <th>Exit</th>
                  <th>PnL</th>
                  <th>Slippage</th>
                  <th>Hold</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((t, i) => (
                  <tr key={t.id || `${t.ts}-${i}`}>
                    <td className="num">{fmtTs(t.ts)}</td>
                    <td className="num">{t.pair}</td>
                    <td className={`num ${dirClass(t.direction)}`}>
                      {t.direction > 0 ? <ArrowUp size={14} /> : t.direction < 0 ? <ArrowDown size={14} /> : "—"}
                    </td>
                    <td className="num">{fmtNum(t.size)}</td>
                    <td className="num">{fmtNum(t.entry_price, 5)}</td>
                    <td className="num">{fmtNum(t.exit_price, 5)}</td>
                    <td>
                      <div className="pnl-cell">
                        <span className={`num ${t.pnl > 0 ? "pos" : t.pnl < 0 ? "neg" : ""}`}>{t.pnl >= 0 ? "+" : ""}{fmtNum(t.pnl)}</span>
                        <div className={`pnl-bar ${t.pnl >= 0 ? "pos" : "neg"}`} style={{ width: `${Math.min(100, Math.abs(t.pnl) / maxAbs * 100)}%` }} />
                      </div>
                    </td>
                    <td className="num">{fmtNum(t.slippage_pips, 2)}</td>
                    <td className="num">{fmtHold(t.hold_steps)}</td>
                    <td>{t.exit_reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-head">
          <h3>Order Log</h3>
        </div>
        {orders.length === 0 ? (
          <div className="empty">No orders yet.</div>
        ) : (
          <div className="scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Dir</th>
                  <th>Type</th>
                  <th>Size</th>
                  <th>Fill</th>
                  <th>Status</th>
                  <th>Slippage</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>
                {orders.slice(0, 100).map((o, i) => {
                  const d = o.data
                  const dir = Number(d.direction ?? 0)
                  return (
                    <tr key={i}>
                      <td className="num">{fmtTs(o.ts)}</td>
                      <td className={`num ${dirClass(d.direction)}`}>
                        {dir > 0 ? <ArrowUp size={14} /> : dir < 0 ? <ArrowDown size={14} /> : "—"}
                      </td>
                      <td>{String(d.order_type ?? "—")}</td>
                      <td className="num">{fmtNum(d.size)}</td>
                      <td className="num">{fmtNum(d.fill_price, 5)}</td>
                      <td>{String(d.status ?? "—")}</td>
                      <td className="num">{fmtNum(d.slippage_pips, 2)}</td>
                      <td className="num">{fmtNum(d.latency_us, 0)}µs</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
