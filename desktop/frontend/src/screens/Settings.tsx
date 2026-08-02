import { useEffect, useMemo, useState } from "react"
import type { EngineHealth } from "../../electron/ipc"
import { useEvents } from "../hooks"

interface Cfg {
  account?: { balance?: number }
  pairs?: string[]
  simulation?: { pair?: string; god_mode?: boolean }
  risk?: {
    sizing?: { method?: string; kelly_fraction?: number; max_account_risk_pct?: number }
    circuit_breakers?: { daily_drawdown_limit?: number; weekly_drawdown_limit?: number; monthly_drawdown_limit?: number }
  }
  execution?: { broker?: string }
}

const DEFAULT_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF"]

function Section({ title, children }: { title: string; children: React.ReactNode }): React.JSX.Element {
  return (
    <div className="card">
      <h3 className="section-title">{title}</h3>
      <div className="section-body">{children}</div>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }): React.JSX.Element {
  return (
    <label className="field muted">
      {label}
      {children}
    </label>
  )
}

export default function Settings(): React.JSX.Element {
  const [cfg, setCfg] = useState<Cfg>({})
  const [health, setHealth] = useState<EngineHealth | null>(null)
  const [engineUrl, setEngineUrl] = useState("")
  const [balance, setBalance] = useState("")
  const [kelly, setKelly] = useState("")
  const [maxRisk, setMaxRisk] = useState("")
  const [cbDaily, setCbDaily] = useState("")
  const [cbWeekly, setCbWeekly] = useState("")
  const [cbMonthly, setCbMonthly] = useState("")
  const [godMode, setGodMode] = useState(true)
  const [simPair, setSimPair] = useState("EURUSD")
  const [pairs, setPairs] = useState<string[]>([])
  const [minimizeTray, setMinimizeTray] = useState(false)
  const [updatesEnabled, setUpdatesEnabled] = useState(false)
  const [updatesFeed, setUpdatesFeed] = useState("")
  const [saved, setSaved] = useState<string | null>(null)
  const [preparing, setPreparing] = useState(false)
  const progress = useEvents("progress")

  const prepLog = useMemo(
    () => progress.filter((p) => typeof p.data.message === "string").map((p) => String(p.data.message)).slice(0, 20),
    [progress]
  )

  const refreshHealth = (): void => {
    void window.api.engine.health().then(setHealth).catch(() => setHealth(null))
  }

  useEffect(() => {
    void window.api.engine.getUrl().then(setEngineUrl)
    void window.api.config
      .get()
      .then((c) => {
        const cfg0 = c as Cfg
        setCfg(cfg0)
        const acct = cfg0.account
        if (typeof acct?.balance === "number") setBalance(String(acct.balance))
        const r = cfg0.risk
        if (typeof r?.sizing?.kelly_fraction === "number") setKelly(String(r.sizing.kelly_fraction))
        if (typeof r?.sizing?.max_account_risk_pct === "number") setMaxRisk(String(r.sizing.max_account_risk_pct))
        const cb = r?.circuit_breakers
        if (typeof cb?.daily_drawdown_limit === "number") setCbDaily(String(cb.daily_drawdown_limit * 100))
        if (typeof cb?.weekly_drawdown_limit === "number") setCbWeekly(String(cb.weekly_drawdown_limit * 100))
        if (typeof cb?.monthly_drawdown_limit === "number") setCbMonthly(String(cb.monthly_drawdown_limit * 100))
        if (Array.isArray(cfg0.pairs)) setPairs(cfg0.pairs as string[])
        setSimPair(String(cfg0.simulation?.pair ?? "EURUSD"))
        setGodMode(cfg0.simulation?.god_mode !== false)
      })
      .catch(() => undefined)
    refreshHealth()
  }, [])

  useEffect(() => {
    void window.api.app.getMinimizeTray().then(setMinimizeTray).catch(() => undefined)
    void window.api.app
      .getUpdates()
      .then((u) => {
        setUpdatesEnabled(u.enabled)
        setUpdatesFeed(u.feedUrl)
      })
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    const active = progress.some((p) => String(p.data.message ?? "").includes("Data preparation complete"))
    if (active) setPreparing(false)
  }, [progress])

  const onSaveUrl = async (): Promise<void> => {
    setSaved(null)
    const res = await window.api.engine.setUrl(engineUrl)
    if (res.ok) {
      setEngineUrl(res.url)
      setSaved("Engine URL saved")
      refreshHealth()
    }
  }

  const onSaveAccount = async (): Promise<void> => {
    const v = Number(balance)
    if (!Number.isFinite(v) || v <= 0) return
    setSaved(null)
    await window.api.config.set({ account: { balance: v } })
    setSaved("Account saved")
  }

  const onSaveMarket = async (): Promise<void> => {
    setSaved(null)
    await window.api.config.set({ pairs, simulation: { pair: simPair } })
    setSaved("Market saved")
  }

  const onSaveRisk = async (): Promise<void> => {
    setSaved(null)
    await window.api.config.set({
      risk: {
        sizing: { kelly_fraction: Number(kelly), max_account_risk_pct: Number(maxRisk) },
        circuit_breakers: {
          daily_drawdown_limit: Number(cbDaily) / 100,
          weekly_drawdown_limit: Number(cbWeekly) / 100,
          monthly_drawdown_limit: Number(cbMonthly) / 100
        }
      }
    })
    setSaved("Risk saved")
  }

  const onSaveGodMode = async (v: boolean): Promise<void> => {
    setGodMode(v)
    setSaved(null)
    await window.api.config.set({ simulation: { god_mode: v } })
    setSaved(`God Mode ${v ? "enabled" : "disabled"} — applies next run`)
  }

  const togglePair = (p: string): void => {
    setPairs((prev) => (prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p]))
  }

  const onPrepare = async (): Promise<void> => {
    setPreparing(true)
    setSaved(null)
    await window.api.data.prepare()
    setSaved("Data preparation started — see progress below")
  }

  const onToggleTray = async (v: boolean): Promise<void> => {
    setMinimizeTray(v)
    setSaved(null)
    await window.api.app.setMinimizeTray(v)
    setSaved(v ? "Close minimizes to tray" : "Close quits the app")
  }

  const onReRunOnboarding = async (): Promise<void> => {
    await window.api.app.setOnboarded(false)
    window.location.reload()
  }

  const onSaveUpdates = async (): Promise<void> => {
    setSaved(null)
    await window.api.app.setUpdates({ enabled: updatesEnabled, feedUrl: updatesFeed })
    setSaved(updatesEnabled ? "Auto-update enabled — applies on next launch" : "Auto-update disabled")
  }

  const ready = health?.data_ready === true
  const connected = health != null
  const allPairs = useMemo(() => [...new Set([...DEFAULT_PAIRS, ...pairs])].sort(), [pairs])

  return (
    <div className="grid cols-2">
      <Section title="Engine">
        <Field label="ENGINE URL">
          <input className="input mono" value={engineUrl} onChange={(e) => setEngineUrl(e.target.value)} />
        </Field>
        <button className="btn" onClick={() => void onSaveUrl()}>Connect</button>
        <div style={{ fontSize: 12 }}>
          <span className={`status-dot ${connected ? "ok" : "bad"}`} />
          {connected ? `Engine ${health!.status} · v${String(health!.version)}` : "Engine offline"}
          {connected && ` · data ${ready ? "ready" : "not prepared"}`}
        </div>
        <div className="divider" />
        <div className="row-between">
          <span className="muted" style={{ fontSize: 12 }}>Broker</span>
          <span className="num">{String(cfg.execution?.broker ?? "paper")}</span>
        </div>
        <div className="row-between">
          <span className="muted" style={{ fontSize: 12 }}>God Mode</span>
          <button className={`toggle ${godMode ? "on" : ""}`} onClick={() => void onSaveGodMode(!godMode)}>
            <span className="knob" />
          </button>
        </div>
        <div className="hint muted">God Mode enables neural/god-mode components. Applies on the next run.</div>
      </Section>

      <Section title="Account">
        <Field label="INITIAL BALANCE (USD)">
          <input className="input mono" type="number" value={balance} onChange={(e) => setBalance(e.target.value)} />
        </Field>
        <button className="btn" onClick={() => void onSaveAccount()}>Save</button>
      </Section>

      <Section title="Market">
        <div className="checks">
          {allPairs.map((p) => (
            <label className="check" key={p}>
              <input type="checkbox" checked={pairs.includes(p)} onChange={() => togglePair(p)} />
              <span className="mono">{p}</span>
            </label>
          ))}
        </div>
        <Field label="SIM PAIR">
          <select className="input mono" value={simPair} onChange={(e) => setSimPair(e.target.value)}>
            {allPairs.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </Field>
        <button className="btn" onClick={() => void onSaveMarket()}>Save</button>
      </Section>

      <Section title="Risk">
        <Field label="KELLY FRACTION">
          <input className="input mono" type="number" step={0.01} value={kelly} onChange={(e) => setKelly(e.target.value)} />
        </Field>
        <Field label="MAX ACCOUNT RISK %">
          <input className="input mono" type="number" step={0.001} value={maxRisk} onChange={(e) => setMaxRisk(e.target.value)} />
        </Field>
        <div className="divider" />
        <Field label="DAILY DRAWDOWN LIMIT %">
          <input className="input mono" type="number" step={0.5} value={cbDaily} onChange={(e) => setCbDaily(e.target.value)} />
        </Field>
        <Field label="WEEKLY DRAWDOWN LIMIT %">
          <input className="input mono" type="number" step={0.5} value={cbWeekly} onChange={(e) => setCbWeekly(e.target.value)} />
        </Field>
        <Field label="MONTHLY DRAWDOWN LIMIT %">
          <input className="input mono" type="number" step={0.5} value={cbMonthly} onChange={(e) => setCbMonthly(e.target.value)} />
        </Field>
        <button className="btn" onClick={() => void onSaveRisk()}>Save</button>
      </Section>

      <Section title="Data">
        <div className="row-between">
          <span className="muted" style={{ fontSize: 12 }}>Status</span>
          <span>{ready ? "Ready" : "Not prepared"}</span>
        </div>
        <button className="btn ghost" disabled={ready || preparing} onClick={() => void onPrepare()}>
          {ready ? "Data ready" : preparing ? "Preparing…" : "Prepare Data"}
        </button>
        {prepLog.length > 0 && (
          <div className="prep-log mono">
            {prepLog.map((m, i) => (
              <div key={i} className="muted">{m}</div>
            ))}
          </div>
        )}
      </Section>

        <Section title="App">
          <div className="row-between">
            <span className="muted" style={{ fontSize: 12 }}>Minimize to tray on close</span>
            <button className={`toggle ${minimizeTray ? "on" : ""}`} onClick={() => void onToggleTray(!minimizeTray)}>
              <span className="knob" />
            </button>
          </div>
          <div className="hint muted">When off, closing the window quits FOREX DESK.</div>
          <div className="divider" />
          <button className="btn ghost" onClick={() => void onReRunOnboarding()}>
            Re-run onboarding
          </button>
        </Section>

        <Section title="Updates">
          <div className="row-between">
            <span className="muted" style={{ fontSize: 12 }}>Enable auto-update</span>
            <button className={`toggle ${updatesEnabled ? "on" : ""}`} onClick={() => setUpdatesEnabled(!updatesEnabled)}>
              <span className="knob" />
            </button>
          </div>
          <Field label="FEED URL">
            <input
              className="input mono"
              value={updatesFeed}
              onChange={(e) => setUpdatesFeed(e.target.value)}
              placeholder="https://example.com/updates"
              disabled={!updatesEnabled}
            />
          </Field>
          <button className="btn" disabled={!updatesEnabled} onClick={() => void onSaveUpdates()}>
            Save
          </button>
          <div className="hint muted">
            {updatesEnabled
              ? "Checks for updates on launch against the feed. Requires a generic electron-updater feed."
              : "Disabled by default. Enable to check for updates on launch."}
          </div>
        </Section>

      {saved && (
        <div className="saved muted" style={{ gridColumn: "1 / -1" }}>
          {saved}
        </div>
      )}
    </div>
  )
}
