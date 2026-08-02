import { useMemo, useState } from "react"
import { ArrowUp, ArrowDown } from "lucide-react"
import { useEvents } from "../hooks"
import { dirClass, fmtNum, fmtPct, fmtTs } from "../format"

export default function Signals(): React.JSX.Element {
  const signals = useEvents("signal")
  const [regime, setRegime] = useState<string>("all")
  const [minConf, setMinConf] = useState<number>(0)

  const regimes = useMemo(() => {
    const s = new Set<string>()
    for (const sig of signals) {
      const r = sig.data.regime
      if (r != null) s.add(String(r))
    }
    return [...s].sort()
  }, [signals])

  const filtered = useMemo(
    () =>
      signals.filter((s) => {
        const d = s.data
        if (regime !== "all" && String(d.regime ?? "") !== regime) return false
        if (minConf > 0 && Number(d.confidence ?? 0) < minConf) return false
        return true
      }),
    [signals, regime, minConf]
  )

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="card">
        <div className="card-head">
          <h3>Signal Stream</h3>
          <div className="filters">
            <select className="input" value={regime} onChange={(e) => setRegime(e.target.value)}>
              <option value="all">All regimes</option>
              {regimes.map((r) => (
                <option key={r} value={r}>Regime {r}</option>
              ))}
            </select>
            <label className="muted filter-num">
              Conf ≥
              <input
                className="input num"
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={minConf}
                onChange={(e) => setMinConf(Number(e.target.value) || 0)}
                style={{ width: 70 }}
              />
            </label>
          </div>
        </div>

        {signals.length === 0 ? (
          <div className="empty">No signals yet — start a simulation.</div>
        ) : filtered.length === 0 ? (
          <div className="empty">No signals match the current filters.</div>
        ) : (
          <div className="scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Dir</th>
                  <th>Magnitude</th>
                  <th>Confidence</th>
                  <th>Uncertainty</th>
                  <th>Regime</th>
                  <th style={{ width: 260 }}>Sub-model agreement</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((s, i) => {
                  const d = s.data
                  const subs = (d.sub_models as Record<string, number>) ?? {}
                  const dir = Number(d.direction ?? 0)
                  return (
                    <tr key={i}>
                      <td className="num">{fmtTs(s.ts)}</td>
                      <td className={`num ${dirClass(d.direction)}`}>
                        {dir > 0 ? <ArrowUp size={16} /> : dir < 0 ? <ArrowDown size={16} /> : "—"}
                      </td>
                      <td className="num">{fmtNum(d.magnitude, 3)}</td>
                      <td className="num">{fmtPct((d.confidence as number) * 100)}</td>
                      <td className="num">{fmtNum(d.uncertainty, 3)}</td>
                      <td className="num">{String(d.regime ?? "—")}</td>
                      <td>
                        <div className="sub-models">
                          {Object.entries(subs).map(([k, v]) => (
                            <div className="sub-model" key={k}>
                              <span className="sub-label muted">{k}</span>
                              <div className="sub-bar">
                                <div className="sub-fill" style={{ width: `${Math.max(0, Math.min(1, Number(v))) * 100}%` }} />
                              </div>
                              <span className="sub-val num">{fmtNum(v, 2)}</span>
                            </div>
                          ))}
                        </div>
                      </td>
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
