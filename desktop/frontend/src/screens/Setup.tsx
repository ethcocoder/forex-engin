import { useEffect, useState } from "react"
import type { EngineHealth } from "../../electron/ipc"

const DEFAULT_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF"]

interface Config {
  account?: { balance?: number }
  pairs?: string[]
  simulation?: { pair?: string }
}

const STEPS = [
  { key: "engine", title: "Engine", hint: "Point FOREX DESK at your engine service." },
  { key: "account", title: "Account", hint: "Set the starting account balance." },
  { key: "market", title: "Market", hint: "Choose the instruments to simulate." }
]

export default function Setup({ onDone }: { onDone: () => void }): React.JSX.Element {
  const [step, setStep] = useState(0)
  const [url, setUrl] = useState("")
  const [urlTouched, setUrlTouched] = useState(false)
  const [health, setHealth] = useState<EngineHealth | null | "checking">(null)
  const [engineBin, setEngineBin] = useState<string | null>(null)
  const [balance, setBalance] = useState("")
  const [pairs, setPairs] = useState<string[]>(DEFAULT_PAIRS)
  const [simPair, setSimPair] = useState("EURUSD")
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    void window.api.engine.getUrl().then(setUrl)
    void window.api.engine
      .detectBinary()
      .then((r) => setEngineBin(r.found ? r.path ?? "engine binary" : null))
      .catch(() => undefined)
    void window.api.config
      .get()
      .then((c) => {
        const cfg = c as Config
        if (typeof cfg.account?.balance === "number") setBalance(String(cfg.account.balance))
        if (Array.isArray(cfg.pairs) && cfg.pairs.length > 0) setPairs(cfg.pairs as string[])
        setSimPair(String(cfg.simulation?.pair ?? "EURUSD"))
      })
      .catch(() => undefined)
  }, [])

  const checkEngine = async (): Promise<void> => {
    setHealth("checking")
    try {
      const h = await window.api.engine.health()
      setHealth(h)
    } catch {
      setHealth(null)
    }
  }

  const next = (): void => {
    setErr(null)
    if (step === 0 && !urlTouched) {
      // engine step: run a connectivity check on entry before continuing
      void checkEngine()
    }
    setStep((s) => Math.min(s + 1, STEPS.length - 1))
  }

  const back = (): void => setStep((s) => Math.max(s - 1, 0))

  const finish = async (): Promise<void> => {
    setSaving(true)
    setErr(null)
    try {
      const newUrl = url.trim().replace(/\/+$/, "")
      const cur = await window.api.engine.getUrl()
      if (newUrl && newUrl !== cur) {
        const res = await window.api.engine.setUrl(newUrl)
        if (!res.ok) throw new Error(`Failed to set engine URL: ${res.url}`)
      }
      const bal = Number(balance)
      if (!Number.isFinite(bal) || bal <= 0) throw new Error("Enter a valid initial balance.")
      await window.api.config.set({
        account: { balance: bal },
        pairs,
        simulation: { pair: simPair }
      })
      await window.api.app.setOnboarded(true)
      onDone()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      setSaving(false)
    }
  }

  const allPairs = [...new Set([...DEFAULT_PAIRS, ...pairs])].sort()
  const togglePair = (p: string): void =>
    setPairs((prev) => (prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p]))

  return (
    <div className="setup">
      <div className="setup-card">
        <div className="brand setup-brand wordmark">
          FOREX<span className="b">DESK</span>
        </div>
        <p className="setup-sub muted">First-run setup</p>

        <div className="setup-steps">
          {STEPS.map((s, i) => (
            <div key={s.key} className={`setup-step ${i === step ? "active" : i < step ? "done" : ""}`}>
              <span className="dot">{i < step ? "✓" : i + 1}</span>
              <span className="name">{s.title}</span>
            </div>
          ))}
        </div>

        <div className="setup-body">
          {step === 0 && (
            <>
              <p className="muted setup-hint">{STEPS[0]!.hint}</p>
              <label className="field muted">
                ENGINE URL
                <input
                  className="input mono"
                  value={url}
                  onChange={(e) => {
                    setUrl(e.target.value)
                    setUrlTouched(true)
                  }}
                  placeholder="http://127.0.0.1:8737"
                />
              </label>
              <div className="row-between">
                <button className="btn ghost" onClick={() => void checkEngine()}>
                  Test connection
                </button>
                <span className="muted" style={{ fontSize: 12 }}>
                  {health === "checking" ? "Checking…" : health === null ? "" : (
                    <span className={health.data_ready ? "ok" : "warn"}>
                      {health.status} · data {health.data_ready ? "ready" : "not prepared"}
                    </span>
                  )}
                </span>
              </div>
              {health === null && (
                <div className="setup-hint muted" style={{ marginTop: 14, fontSize: 12 }}>
                  {engineBin ? (
                    <>Engine binary found at <code className="mono">{engineBin}</code> — start it, then Connect.</>
                  ) : (
                    <>Run the engine locally: <code className="mono">python engine-server/server.py</code> (port 8737)</>
                  )}
                </div>
              )}
            </>
          )}

          {step === 1 && (
            <>
              <p className="muted setup-hint">{STEPS[1]!.hint}</p>
              <label className="field muted">
                INITIAL BALANCE (USD)
                <input className="input mono" type="number" value={balance} onChange={(e) => setBalance(e.target.value)} />
              </label>
            </>
          )}

          {step === 2 && (
            <>
              <p className="muted setup-hint">{STEPS[2]!.hint}</p>
              <div className="checks">
                {allPairs.map((p) => (
                  <label className="check" key={p}>
                    <input type="checkbox" checked={pairs.includes(p)} onChange={() => togglePair(p)} />
                    <span className="mono">{p}</span>
                  </label>
                ))}
              </div>
              <label className="field muted">
                SIM PAIR
                <select className="input mono" value={simPair} onChange={(e) => setSimPair(e.target.value)}>
                  {allPairs.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </label>
            </>
          )}
        </div>

        {err && <div className="setup-err">{err}</div>}

        <div className="setup-nav">
          {step > 0 && (
            <button className="btn ghost" onClick={back} disabled={saving}>
              Back
            </button>
          )}
          <div className="spacer" />
          {step < STEPS.length - 1 ? (
            <button className="btn primary" onClick={next}>
              Next
            </button>
          ) : (
            <button className="btn primary" onClick={() => void finish()} disabled={saving || pairs.length === 0}>
              {saving ? "Saving…" : "Get Started"}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
