export function dirArrow(d: unknown): string {
  if (d == null) return "·"
  const n = Number(d)
  if (n > 0) return "▲"
  if (n < 0) return "▼"
  return "·"
}

export function dirClass(d: unknown): string {
  const n = Number(d)
  if (n > 0) return "pos"
  if (n < 0) return "neg"
  return ""
}

export function fmtNum(v: unknown, digits = 2): string {
  if (v == null || Number.isNaN(Number(v))) return "—"
  return Number(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function fmtPct(v: unknown, digits = 1): string {
  if (v == null || Number.isNaN(Number(v))) return "—"
  return `${Number(v).toFixed(digits)}%`
}

export function fmtTs(ts: unknown): string {
  if (ts == null) return "—"
  const s = Number(ts)
  if (!Number.isFinite(s) || s <= 0) return "—"
  return new Date(s * 1000).toLocaleString(undefined, {
    hour12: false,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  })
}

export function fmtHold(steps: unknown): string {
  if (steps == null) return "—"
  return `${steps} bars`
}
