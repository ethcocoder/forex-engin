import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Bell,
  Bot,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleDollarSign,
  Clock3,
  Command,
  Cpu,
  FileText,
  Gauge,
  Globe2,
  Languages,
  LockKeyhole,
  Menu,
  Moon,
  Play,
  Radio,
  RefreshCw,
  Rocket,
  Settings2,
  ShieldCheck,
  Sun,
  TerminalSquare,
  TrendingUp,
  WalletCards,
  X,
  Zap,
} from 'lucide-react'

type Language = 'en' | 'am'
type Theme = 'dark' | 'light'
type Section = 'overview' | 'engine' | 'risk' | 'reports' | 'settings'

type EngineStatus = {
  connected: boolean
  mode: string
  version: string
  heartbeat: string
  message: string
}

declare global {
  interface Window {
    electronAPI?: {
      getStatus: () => Promise<EngineStatus>
      runChaos: () => Promise<{ ok: boolean; output: string }>
    }
  }
}

const copy = {
  en: {
    eyebrow: 'ELITE10X · OPERATIONS DESK',
    title: 'Command the engine with clarity.',
    subtitle: 'A bilingual control plane for signal intelligence, risk gates, paper execution, and institutional reporting.',
    overview: 'Overview',
    engine: 'Engine room',
    risk: 'Risk guardrails',
    reports: 'Reports',
    settings: 'Settings',
    livePaper: 'Paper environment',
    connected: 'Broker link pending',
    refresh: 'Refresh status',
    runChaos: 'Run chaos test',
    running: 'Running test…',
    thisMonth: 'This month',
    equity: 'Paper equity',
    pnl: 'Net P&L',
    drawdown: 'Max drawdown',
    winRate: 'Win rate',
    systemHealth: 'System health',
    allSystems: 'All control loops nominal',
    signalQuality: 'Signal quality',
    confidence: 'Confidence',
    throughput: 'Throughput',
    trades: 'Trades',
    latency: 'Tick latency',
    safety: 'Safety posture',
    protected: 'Protected',
    gateCopy: 'Uncertainty, spread, and drawdown gates are active. Live orders remain disabled until a broker is explicitly configured.',
    session: 'Session telemetry',
    pair: 'Pair',
    bias: 'Bias',
    price: 'Mid price',
    change: '24h change',
    long: 'Long',
    short: 'Short',
    neutral: 'Neutral',
    commandDeck: 'Command deck',
    commandCopy: 'Use the control plane to observe the system. Paper mode is enabled by default.',
    launchPaper: 'Launch paper session',
    openReport: 'Open latest report',
    configureBroker: 'Configure broker',
    recentEvents: 'Recent events',
    event1: 'Uncertainty gate rejected 1,324 ambiguous ticks.',
    event2: 'Daily drawdown breaker is armed at 2.0%.',
    event3: 'C++ engine heartbeat verified from local build.',
    event4: 'Bilingual workspace initialized.',
    demoNotice: 'Desktop preview mode: broker API is not connected.',
    safeByDefault: 'Safe by default',
    paperMode: 'PAPER MODE',
    language: 'Language',
    theme: 'Theme',
    light: 'Light',
    dark: 'Dark',
    noLive: 'Live execution disabled',
    noLiveCopy: 'Connect and validate a demo broker before enabling any live workflow.',
    testOutput: 'Latest test output',
    noTest: 'No chaos test run in this session.',
    controls: 'Controls',
    autoRefresh: 'Auto-refresh',
    enabled: 'Enabled',
    disabled: 'Disabled',
    appVersion: 'App version',
  },
  am: {
    eyebrow: 'ELITE10X · የስራ ማዕከል',
    title: 'ሞተሩን በግልጽነት ይቆጣጠሩ።',
    subtitle: 'ለሲግናል፣ ለአደጋ ቁጥጥር፣ ለፔፐር ንግድ እና ለሪፖርት የተዘጋጀ ባለሁለት ቋንቋ መቆጣጠሪያ።',
    overview: 'አጠቃላይ እይታ',
    engine: 'የሞተር ክፍል',
    risk: 'የአደጋ መከላከያ',
    reports: 'ሪፖርቶች',
    settings: 'ቅንብሮች',
    livePaper: 'የፔፐር አካባቢ',
    connected: 'የባንክ ግንኙነት በመጠባበቅ ላይ',
    refresh: 'ሁኔታን አድስ',
    runChaos: 'የአደጋ ሙከራ አስኪድ',
    running: 'በመስራት ላይ…',
    thisMonth: 'በዚህ ወር',
    equity: 'የፔፐር ካፒታል',
    pnl: 'ጠቅላላ P&L',
    drawdown: 'ከፍተኛ መቀነስ',
    winRate: 'የማሸነፍ መጠን',
    systemHealth: 'የስርዓት ጤና',
    allSystems: 'ሁሉም ቁጥጥሮች መደበኛ ናቸው',
    signalQuality: 'የሲግናል ጥራት',
    confidence: 'እምነት',
    throughput: 'የስራ ፍጥነት',
    trades: 'ንግዶች',
    latency: 'የቲክ መዘግየት',
    safety: 'የደህንነት ሁኔታ',
    protected: 'የተጠበቀ',
    gateCopy: 'የእርግጠኝነት፣ የስፕሬድ እና የመቀነስ መከላከያዎች ንቁ ናቸው። ባንክ በግልጽ እስካልተዘጋጀ ድረስ የቀጥታ ትዕዛዞች ተዘግተዋል።',
    session: 'የሴሽን መረጃ',
    pair: 'ጥንድ',
    bias: 'አቅጣጫ',
    price: 'መካከለኛ ዋጋ',
    change: 'የ24 ሰዓት ለውጥ',
    long: 'ግዛ',
    short: 'ሽጥ',
    neutral: 'ገለልተኛ',
    commandDeck: 'የትዕዛዝ መድረክ',
    commandCopy: 'ስርዓቱን ለመመልከት ይጠቀሙ። የፔፐር ሁኔታ በነባሪነት ንቁ ነው።',
    launchPaper: 'የፔፐር ሴሽን ክፈት',
    openReport: 'የቅርብ ሪፖርት ክፈት',
    configureBroker: 'ባንክ አዘጋጅ',
    recentEvents: 'የቅርብ ክስተቶች',
    event1: '1,324 ግልጽ ያልሆኑ ቲኮች በእርግጠኝነት መከላከያ ተቀባይነት አላገኙም።',
    event2: 'የዕለት መቀነስ መከላከያ በ2.0% ተዘጋጅቷል።',
    event3: 'የC++ ሞተር ልብ ምት ተረጋግጧል።',
    event4: 'ባለሁለት ቋንቋ የስራ ቦታ ተጀምሯል።',
    demoNotice: 'የዴስክቶፕ ቅድመ እይታ፦ የባንክ API አልተገናኘም።',
    safeByDefault: 'በነባሪ ደህንነቱ የተጠበቀ',
    paperMode: 'የፔፐር ሁኔታ',
    language: 'ቋንቋ',
    theme: 'ገጽታ',
    light: 'ብርሃን',
    dark: 'ጨለማ',
    noLive: 'የቀጥታ ንግድ ተዘግቷል',
    noLiveCopy: 'ቀጥታ ሂደት ከመክፈትዎ በፊት የዴሞ ባንክ ያገናኙ እና ያረጋግጡ።',
    testOutput: 'የቅርብ ሙከራ ውጤት',
    noTest: 'በዚህ ሴሽን ሙከራ አልተካሄደም።',
    controls: 'ቁጥጥሮች',
    autoRefresh: 'ራስ-ሰር ማደስ',
    enabled: 'ንቁ',
    disabled: 'ዝግ',
    appVersion: 'የመተግበሪያ ስሪት',
  },
} as const

type CopyKey = keyof typeof copy.en

const navItems: Array<{ id: Section; icon: typeof Activity; key: CopyKey }> = [
  { id: 'overview', icon: BarChart3, key: 'overview' },
  { id: 'engine', icon: BrainCircuit, key: 'engine' },
  { id: 'risk', icon: ShieldCheck, key: 'risk' },
  { id: 'reports', icon: FileText, key: 'reports' },
  { id: 'settings', icon: Settings2, key: 'settings' },
]

const sessionRows = [
  { pair: 'EUR/USD', bias: 'long', price: '1.0842', change: '+0.84%', color: 'green' },
  { pair: 'GBP/USD', bias: 'short', price: '1.2676', change: '-0.31%', color: 'red' },
  { pair: 'USD/JPY', bias: 'long', price: '151.44', change: '+0.22%', color: 'green' },
  { pair: 'AUD/USD', bias: 'neutral', price: '0.6548', change: '+0.04%', color: 'slate' },
]

const sparklinePoints = '0,71 18,65 36,69 54,52 72,58 90,48 108,54 126,38 144,43 162,25 180,31 198,15 216,20 234,8 252,12 270,0'

function App() {
  const [language, setLanguage] = useState<Language>('en')
  const [theme, setTheme] = useState<Theme>('dark')
  const [section, setSection] = useState<Section>('overview')
  const [mobileNav, setMobileNav] = useState(false)
  const [running, setRunning] = useState(false)
  const [testOutput, setTestOutput] = useState('')
  const [status, setStatus] = useState<EngineStatus>({
    connected: false,
    mode: 'PAPER',
    version: 'elite10x-pr / desktop-control-plane',
    heartbeat: new Date().toISOString(),
    message: 'Broker credentials are not configured. Paper mode is safe by default.',
  })
  const t = (key: CopyKey) => copy[language][key]

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.documentElement.lang = language === 'am' ? 'am' : 'en'
  }, [theme, language])

  useEffect(() => {
    window.electronAPI?.getStatus().then(setStatus).catch(() => undefined)
  }, [])

  const displaySection = useMemo(() => {
    if (section === 'overview') return t('overview')
    if (section === 'engine') return t('engine')
    if (section === 'risk') return t('risk')
    if (section === 'reports') return t('reports')
    return t('settings')
  }, [section, language])

  async function runChaosTest() {
    setRunning(true)
    setTestOutput('')
    try {
      if (window.electronAPI) {
        const result = await window.electronAPI.runChaos()
        setTestOutput(result.output || 'Chaos test returned no output.')
      } else {
        await new Promise((resolve) => setTimeout(resolve, 1100))
        setTestOutput('Desktop preview fallback\n10,000 ticks · 0.96 ms · 10 black swan deflections · 1,324 uncertainty rejections')
      }
    } finally {
      setRunning(false)
    }
  }

  function selectSection(next: Section) {
    setSection(next)
    setMobileNav(false)
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
        <div className="brand-lockup">
          <div className="brand-mark"><Command size={18} strokeWidth={2.5} /></div>
          <div>
            <div className="brand-name">FOREX <span>ENGIN</span></div>
            <div className="brand-caption">ELITE10X CONTROL</div>
          </div>
          <button className="icon-btn sidebar-close" onClick={() => setMobileNav(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>

        <div className="side-status">
          <span className="status-dot" />
          <div>
            <div className="side-status-title">{t('livePaper')}</div>
            <div className="side-status-copy">{status.connected ? 'Connected' : t('connected')}</div>
          </div>
          <Radio size={15} className="muted-icon" />
        </div>

        <nav className="side-nav" aria-label="Primary navigation">
          <div className="side-nav-label">WORKSPACE</div>
          {navItems.map(({ id, icon: Icon, key }) => (
            <button key={id} className={`nav-item ${section === id ? 'active' : ''}`} onClick={() => selectSection(id)}>
              <Icon size={18} />
              <span>{t(key)}</span>
              {id === 'risk' && <span className="nav-alert">2</span>}
              {section === id && <ChevronRight size={16} className="nav-chevron" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="build-card">
            <div className="build-card-top"><Cpu size={16} /><span>{t('safeByDefault')}</span></div>
            <div className="build-progress"><span /></div>
            <div className="build-meta"><span>{t('paperMode')}</span><span>v0.1.0</span></div>
          </div>
          <div className="sidebar-footer"><LockKeyhole size={13} /> {t('noLive')}</div>
        </div>
      </aside>

      {mobileNav && <button className="mobile-backdrop" onClick={() => setMobileNav(false)} aria-label="Close navigation" />}

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-btn mobile-menu" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={20} /></button>
            <div className="breadcrumb"><span>CONTROL CENTER</span><ChevronRight size={14} /><strong>{displaySection.toUpperCase()}</strong></div>
          </div>
          <div className="topbar-actions">
            <button className="mode-pill" onClick={() => setSection('settings')}><span className="status-dot" /> {t('paperMode')} <ChevronRight size={14} /></button>
            <button className="icon-btn" aria-label="Notifications"><Bell size={18} /><span className="notification-dot" /></button>
            <div className="avatar">AE</div>
          </div>
        </header>

        <div className="page-wrap">
          <section className="hero-row">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> {t('eyebrow')}</div>
              <h1>{t('title')}</h1>
              <p className="hero-subtitle">{t('subtitle')}</p>
            </div>
            <div className="hero-controls">
              <button className="control-btn" onClick={() => setLanguage(language === 'en' ? 'am' : 'en')}><Languages size={16} /> {language === 'en' ? 'አማ' : 'EN'}</button>
              <button className="control-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme">{theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />} {theme === 'dark' ? t('light') : t('dark')}</button>
              <button className="primary-btn" onClick={runChaosTest} disabled={running}><Play size={15} fill="currentColor" /> {running ? t('running') : t('runChaos')}</button>
            </div>
          </section>

          <section className="alert-banner">
            <div className="alert-icon"><AlertTriangle size={17} /></div>
            <div><strong>{t('demoNotice')}</strong><span>{t('noLiveCopy')}</span></div>
            <button className="alert-action" onClick={() => setSection('settings')}>{t('configureBroker')} <ChevronRight size={14} /></button>
          </section>

          <section className="metrics-grid">
            <MetricCard label={t('equity')} value="$100,000" delta="+0.0%" icon={<WalletCards size={17} />} tone="cyan" helper={t('thisMonth')} />
            <MetricCard label={t('pnl')} value="$0.00" delta="PAPER" icon={<CircleDollarSign size={17} />} tone="green" helper={t('thisMonth')} />
            <MetricCard label={t('drawdown')} value="0.00%" delta="SAFE" icon={<ShieldCheck size={17} />} tone="violet" helper="24H" />
            <MetricCard label={t('winRate')} value="—" delta="PENDING" icon={<TrendingUp size={17} />} tone="amber" helper={t('thisMonth')} />
          </section>

          <section className="content-grid">
            <div className="panel chart-panel">
              <div className="panel-heading">
                <div><div className="panel-kicker">EQUITY CURVE</div><h2>{t('equity')}</h2></div>
                <div className="chart-range"><button className="range-active">1D</button><button>1W</button><button>1M</button><button>ALL</button></div>
              </div>
              <div className="chart-summary"><strong>$100,000.00</strong><span className="muted-copy">Awaiting paper fills</span></div>
              <div className="chart-wrap">
                <div className="chart-y-labels"><span>102k</span><span>101k</span><span>100k</span><span>99k</span></div>
                <svg className="equity-chart" viewBox="0 0 270 80" preserveAspectRatio="none" aria-label="Equity chart preview">
                  <defs><linearGradient id="areaGradient" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#55d6c2" stopOpacity=".22" /><stop offset="100%" stopColor="#55d6c2" stopOpacity="0" /></linearGradient></defs>
                  <path d={`M ${sparklinePoints} L270 80 L0 80 Z`} fill="url(#areaGradient)" />
                  <polyline points={sparklinePoints} fill="none" stroke="#55d6c2" strokeWidth="1.8" vectorEffect="non-scaling-stroke" />
                  <circle cx="270" cy="0" r="2.8" fill="#55d6c2" />
                </svg>
                <div className="chart-x-labels"><span>09:00</span><span>12:00</span><span>15:00</span><span>18:00</span></div>
              </div>
            </div>

            <div className="panel health-panel">
              <div className="panel-heading"><div><div className="panel-kicker">RUNTIME TELEMETRY</div><h2>{t('systemHealth')}</h2></div><Activity size={18} className="panel-icon" /></div>
              <div className="health-score"><div className="score-ring"><span>98</span><small>/ 100</small></div><div><strong>{t('allSystems')}</strong><p>Heartbeat {new Date(status.heartbeat).toLocaleTimeString()}</p></div></div>
              <HealthRow label={t('signalQuality')} value="0.94" width="94%" color="cyan" />
              <HealthRow label={t('throughput')} value="1.8M / sec" width="86%" color="violet" />
              <HealthRow label={t('latency')} value="0.96 ms" width="96%" color="green" />
              <div className="health-footer"><span><span className="status-dot" /> {t('protected')}</span><span>0 incidents</span></div>
            </div>
          </section>

          <section className="content-grid lower-grid">
            <div className="panel table-panel">
              <div className="panel-heading"><div><div className="panel-kicker">MARKET WATCH</div><h2>{t('session')}</h2></div><button className="icon-btn small" onClick={() => window.electronAPI?.getStatus().then(setStatus)} aria-label={t('refresh')}><RefreshCw size={15} /></button></div>
              <div className="table-wrap"><table><thead><tr><th>{t('pair')}</th><th>{t('bias')}</th><th>{t('price')}</th><th>{t('change')}</th></tr></thead><tbody>{sessionRows.map((row) => <tr key={row.pair}><td><span className="pair-dot" />{row.pair}</td><td><span className={`bias-pill ${row.bias}`}>{row.bias === 'long' ? <ArrowUpRight size={12} /> : row.bias === 'short' ? <ArrowDownRight size={12} /> : <span className="neutral-dash" />} {row.bias === 'long' ? t('long') : row.bias === 'short' ? t('short') : t('neutral')}</span></td><td className="mono">{row.price}</td><td className={row.color === 'green' ? 'positive' : row.color === 'red' ? 'negative' : 'muted-copy'}>{row.change}</td></tr>)}</tbody></table></div>
            </div>

            <div className="panel command-panel">
              <div className="panel-heading"><div><div className="panel-kicker">OPERATIONS</div><h2>{t('commandDeck')}</h2></div><TerminalSquare size={18} className="panel-icon" /></div>
              <p className="command-copy">{t('commandCopy')}</p>
              <div className="command-actions"><button className="command-primary" onClick={() => setSection('engine')}><Rocket size={15} /> {t('launchPaper')}</button><button className="command-secondary" onClick={() => setSection('reports')}><FileText size={15} /> {t('openReport')}</button></div>
              <div className="command-divider" />
              <div className="command-mini-row"><span><Zap size={14} /> {t('autoRefresh')}</span><strong className="toggle-on">{t('enabled')}</strong></div>
              <div className="command-mini-row"><span><Globe2 size={14} /> {t('appVersion')}</span><strong>0.1.0</strong></div>
            </div>
          </section>

          <section className="bottom-grid">
            <div className="panel events-panel"><div className="panel-heading"><div><div className="panel-kicker">AUDIT TRAIL</div><h2>{t('recentEvents')}</h2></div><Clock3 size={18} className="panel-icon" /></div><div className="events-list"><EventItem icon={<ShieldCheck size={14} />} text={t('event1')} time="2m ago" /><EventItem icon={<Gauge size={14} />} text={t('event2')} time="5m ago" /><EventItem icon={<CheckCircle2 size={14} />} text={t('event3')} time="9m ago" /><EventItem icon={<Languages size={14} />} text={t('event4')} time="12m ago" /></div></div>
            <div className="panel test-panel"><div className="panel-heading"><div><div className="panel-kicker">VALIDATION</div><h2>{t('testOutput')}</h2></div><Bot size={18} className="panel-icon" /></div><pre className={`test-output ${testOutput ? 'has-output' : ''}`}>{testOutput || t('noTest')}</pre></div>
          </section>

          <footer className="page-footer"><span>FOREX ENGIN · ELITE10X CONTROL PLANE</span><span><span className="status-dot" /> {t('safeByDefault')}</span></footer>
        </div>
      </main>
    </div>
  )
}

function MetricCard({ label, value, delta, icon, tone, helper }: { label: string; value: string; delta: string; icon: React.ReactNode; tone: string; helper: string }) {
  return <div className="metric-card"><div className={`metric-icon ${tone}`}>{icon}</div><div className="metric-info"><span>{label}</span><strong>{value}</strong><small className={tone === 'green' ? 'positive' : 'muted-copy'}>{delta} · {helper}</small></div><div className={`metric-line ${tone}`} /></div>
}

function HealthRow({ label, value, width, color }: { label: string; value: string; width: string; color: string }) {
  return <div className="health-row"><div><span>{label}</span><strong>{value}</strong></div><div className="health-track"><span className={color} style={{ width }} /></div></div>
}

function EventItem({ icon, text, time }: { icon: React.ReactNode; text: string; time: string }) {
  return <div className="event-item"><span className="event-icon">{icon}</span><span className="event-text">{text}</span><time>{time}</time></div>
}

export default App
