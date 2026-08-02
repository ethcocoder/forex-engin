import { useEffect, useState } from "react"
import type { TearSheet } from "../../electron/ipc"
import { useLatest } from "../hooks"
import { fmtNum, fmtPct } from "../format"
import Markdown from "../components/Markdown"

function download(filename: string, content: string): void {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export default function Reports(): React.JSX.Element {
  const [sheet, setSheet] = useState<TearSheet | null>(null)
  const [error, setError] = useState<string | null>(null)
  const report = useLatest("report")

  useEffect(() => {
    void window.api.reports
      .tearSheet()
      .then(setSheet)
      .catch((e: Error) => setError(e.message))
  }, [])

  useEffect(() => {
    if (report?.data?.metrics && report.data.markdown) {
      setSheet({ markdown: String(report.data.markdown), metrics: report.data.metrics as Record<string, number> })
    }
  }, [report])

  const metrics = sheet?.metrics ?? {}
  const empty = sheet != null && (sheet.markdown.includes("No completed run") || sheet.markdown.includes("Insufficient data"))

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="card">
        <div className="card-head">
          <h3>Performance Report</h3>
          <div className="filters">
            <button className="btn ghost" disabled={!sheet || empty} onClick={() => sheet && void window.api.reports.export("csv").then((r) => download(r.filename, r.content))}>
              Export CSV
            </button>
            <button className="btn ghost" disabled={!sheet || empty} onClick={() => sheet && void window.api.reports.export("json").then((r) => download(r.filename, r.content))}>
              Export JSON
            </button>
          </div>
        </div>

        {error && <div className="empty" style={{ color: "var(--neg)" }}>Engine unavailable: {error}</div>}
        {!error && !sheet && <div className="empty">Loading report…</div>}
        {!error && sheet && (
          <>
            {!empty && (
              <div className="grid cols-4" style={{ marginBottom: 16 }}>
                {Object.entries(metrics).map(([k, v]) => (
                  <div className="metric" key={k}>
                    <div className="lbl">{k.replace(/_/g, " ")}</div>
                    <div className="val num" style={{ fontSize: 16 }}>
                      {k.includes("pct") || k.includes("rate") || k.includes("sharpe") ? fmtPct(v, 2) : fmtNum(v, 2)}
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div className="md-box">
              <Markdown text={sheet.markdown} />
            </div>
          </>
        )}
      </div>
    </div>
  )
}
