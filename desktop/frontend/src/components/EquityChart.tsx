import { fmtNum } from "../format"

export interface EquityPoint {
  ts: number
  equity: number
}

interface Props {
  points: EquityPoint[]
  height?: number
}

export default function EquityChart({ points, height = 220 }: Props): React.JSX.Element {
  if (points.length === 0) {
    return (
      <div className="empty" style={{ height }}>
        No equity data yet — start a simulation.
      </div>
    )
  }

  const W = 800
  const H = 220
  const PAD = 8
  const pad = 6
  const values = points.map((p) => p.equity)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const y = (v: number): number => PAD + (1 - (v - min) / span) * (H - 2 * PAD)
  const x = (i: number): number => (i / (points.length - 1)) * W
  const last = points[points.length - 1].equity
  const drawdown = ((max - last) / max) * 100
  const paper = "var(--paper)"
  const gray = "var(--gray)"

  const line = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`).join(" ")

  let maxIdx = 0
  for (let i = 1; i < points.length; i++) if (points[i].equity > points[maxIdx].equity) maxIdx = i
  const maxY = y(max)
  const maxX = x(maxIdx)
  const lastX = x(points.length - 1)
  const lastY = y(last)

  return (
    <div className="chart" style={{ height }}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
        <defs>
          <linearGradient id="eqArea" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={paper} stopOpacity="0.12" />
            <stop offset="100%" stopColor={paper} stopOpacity="0" />
          </linearGradient>
        </defs>

        <path d={`${line} L${lastX},${H} L0,${H} Z`} fill="url(#eqArea)" stroke="none" />
        <path d={line} fill="none" stroke={paper} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />

        <path
          d={`M0,${maxY} L${lastX},${maxY} L${lastX},${lastY} L${x(maxIdx)},${lastY} Z`}
          fill={gray}
          opacity="0.12"
          stroke="none"
        />

        <line x1={maxX} y1={maxY} x2={maxX} y2={maxY} stroke={paper} strokeWidth="2" vectorEffect="non-scaling-stroke" />
        <line x1={lastX} y1={lastY} x2={lastX} y2={lastY} stroke={paper} strokeWidth="2" vectorEffect="non-scaling-stroke" />
      </svg>

      <div className="chart-labels">
        <div className="chart-label num">${fmtNum(max)}</div>
        <div className="chart-label num">${fmtNum(last)}</div>
        <div className="chart-label num">${fmtNum(min)}</div>
        <div className={`chart-label num ${drawdown > 0 ? "neg" : "pos"}`}>DD {drawdown.toFixed(1)}%</div>
      </div>
    </div>
  )
}
