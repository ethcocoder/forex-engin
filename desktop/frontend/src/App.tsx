import { useEffect, useState } from "react"
import { HashRouter, Navigate, Route, Routes } from "react-router-dom"
import Layout from "./components/Layout"
import Dashboard from "./screens/Dashboard"
import Signals from "./screens/Signals"
import Trades from "./screens/Trades"
import Reports from "./screens/Reports"
import Settings from "./screens/Settings"
import Setup from "./screens/Setup"
import { useLatest } from "./hooks"

function Banner(): React.JSX.Element | null {
  const engineStatus = useLatest("engine.status")
  const auditConn = useLatest("audit.connection")
  const [health, setHealth] = useState<{ data_ready: boolean } | null>(null)
  const [everConnected, setEverConnected] = useState(false)

  const state = String(engineStatus?.data?.state ?? "unknown")
  const connected = state === "connected"
  const wsConnected = auditConn?.data?.connected === true

  useEffect(() => {
    if (connected) setEverConnected(true)
  }, [connected])

  useEffect(() => {
    void window.api.engine.health().then((h) => setHealth(h)).catch(() => undefined)
  }, [connected])

  if (connected && health?.data_ready) return null

  let msg: string
  let tone: "warn" | "err"
  if (!connected) {
    msg = everConnected && !wsConnected
      ? "Engine connection lost — reconnecting…"
      : "Engine offline — start the engine service, then open Settings → Engine URL."
    tone = "err"
  } else {
    msg = "Engine data not prepared — open Settings → Prepare Data."
    tone = "warn"
  }

  return <div className={`banner ${tone}`}>{msg}</div>
}

export default function App(): React.JSX.Element | null {
  const [onboarded, setOnboarded] = useState<boolean | null>(null)

  useEffect(() => {
    void window.api.app.getOnboarded().then(setOnboarded)
  }, [])

  if (onboarded === null) return null

  if (!onboarded) {
    return (
      <Setup
        onDone={() => {
          setOnboarded(true)
          window.location.hash = "#/"
        }}
      />
    )
  }

  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<><Banner /><Dashboard /></>} />
          <Route path="/signals" element={<><Banner /><Signals /></>} />
          <Route path="/trades" element={<><Banner /><Trades /></>} />
          <Route path="/reports" element={<><Banner /><Reports /></>} />
          <Route path="/settings" element={<><Banner /><Settings /></>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
