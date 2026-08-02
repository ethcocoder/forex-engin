import Database from "better-sqlite3"

export interface TradeRow {
  id: number
  ts: number
  pair: string
  direction: number
  size: number
  entry_price: number
  exit_price: number
  pnl: number
  slippage_pips: number
  hold_steps: number
  exit_reason: string
}

export interface OrderRow {
  id: number
  ts: number
  pair: string
  direction: number
  size: number
  order_type: string
  status: string
  fill_price: number | null
  slippage_pips: number | null
  latency_us: number | null
}

export interface AlertRow {
  id: number
  ts: number
  level: string
  source: string
  code: string
  message: string
}

/**
 * Standalone SQLite audit store. No Electron dependency, so it can be unit
 * tested under plain Node with the same better-sqlite3 build.
 */
export class AuditStore {
  private db: Database.Database

  constructor(dbPath: string) {
    this.db = new Database(dbPath)
    this.db.pragma("journal_mode = WAL")
    this.migrate()
  }

  private migrate(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        pair TEXT NOT NULL,
        direction INTEGER NOT NULL,
        size REAL NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL NOT NULL,
        pnl REAL NOT NULL,
        slippage_pips REAL NOT NULL,
        hold_steps INTEGER NOT NULL,
        exit_reason TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades (ts DESC);
      CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        pair TEXT NOT NULL,
        direction INTEGER NOT NULL,
        size REAL NOT NULL,
        order_type TEXT NOT NULL,
        status TEXT NOT NULL,
        fill_price REAL,
        slippage_pips REAL,
        latency_us REAL
      );
      CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders (ts DESC);
      CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        level TEXT NOT NULL,
        source TEXT NOT NULL,
        code TEXT NOT NULL,
        message TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts (ts DESC);
    `)
  }

  insertTrade(row: Omit<TradeRow, "id">): void {
    this.db
      .prepare(
        `INSERT INTO trades (ts, pair, direction, size, entry_price, exit_price, pnl, slippage_pips, hold_steps, exit_reason)
         VALUES (@ts, @pair, @direction, @size, @entry_price, @exit_price, @pnl, @slippage_pips, @hold_steps, @exit_reason)`
      )
      .run(row as unknown as Record<string, unknown>)
  }

  insertOrder(row: Omit<OrderRow, "id">): void {
    this.db
      .prepare(
        `INSERT INTO orders (ts, pair, direction, size, order_type, status, fill_price, slippage_pips, latency_us)
         VALUES (@ts, @pair, @direction, @size, @order_type, @status, @fill_price, @slippage_pips, @latency_us)`
      )
      .run(row as unknown as Record<string, unknown>)
  }

  insertAlert(row: Omit<AlertRow, "id">): void {
    this.db
      .prepare(
        `INSERT INTO alerts (ts, level, source, code, message)
         VALUES (@ts, @level, @source, @code, @message)`
      )
      .run(row as unknown as Record<string, unknown>)
  }

  recentTrades(limit = 500): TradeRow[] {
    return this.db
      .prepare(`SELECT * FROM trades ORDER BY ts DESC LIMIT ?`)
      .all(limit) as TradeRow[]
  }

  recentOrders(limit = 500): OrderRow[] {
    return this.db
      .prepare(`SELECT * FROM orders ORDER BY ts DESC LIMIT ?`)
      .all(limit) as OrderRow[]
  }

  recentAlerts(limit = 500): AlertRow[] {
    return this.db
      .prepare(`SELECT * FROM alerts ORDER BY ts DESC LIMIT ?`)
      .all(limit) as AlertRow[]
  }

  counts(): { trades: number; orders: number; alerts: number } {
    const one = (t: string): number =>
      (this.db.prepare(`SELECT COUNT(*) AS n FROM ${t}`).get() as { n: number }).n
    return { trades: one("trades"), orders: one("orders"), alerts: one("alerts") }
  }

  prune(keepPerTable = 50000): void {
    const prune = (t: string): void => {
      this.db.prepare(
        `DELETE FROM ${t} WHERE id NOT IN (SELECT id FROM ${t} ORDER BY ts DESC LIMIT ?)`
      ).run(keepPerTable)
    }
    prune("trades")
    prune("orders")
    prune("alerts")
  }

  checkpoint(): void {
    try {
      this.db.pragma("wal_checkpoint(TRUNCATE)")
    } catch {
      /* ignore */
    }
  }

  close(): void {
    this.db.close()
  }
}

const WS_TYPES = new Set(["trade", "order", "alert"])

/**
 * Watches the engine WebSocket and persists audit rows into the store.
 * Reconnects with capped exponential backoff. Never dies — keeps trying.
 */
export class AuditRecorder {
  private ws: WebSocket | null = null
  private stopped = false
  private timer: ReturnType<typeof setTimeout> | null = null
  private backoff = 1000
  private reconnectDelay = 1000

  constructor(
    private wsUrl: string,
    private store: AuditStore,
    private onStatus?: (connected: boolean) => void,
    private onEvent?: (event: Record<string, unknown>) => void
  ) {}

  start(): void {
    this.stopped = false
    this.connect()
  }

  stop(): void {
    this.stopped = true
    if (this.timer) clearTimeout(this.timer)
    this.closeSocket()
  }

  setUrl(url: string): void {
    this.wsUrl = url
    this.reconnectDelay = 1000
    this.closeSocket()
    if (!this.stopped) this.connect()
  }

  private connect(): void {
    if (this.stopped) return
    let ws: WebSocket
    try {
      ws = new WebSocket(this.wsUrl)
    } catch {
      this.scheduleReconnect()
      return
    }
    this.ws = ws
    ws.onopen = () => {
      this.backoff = 1000
      this.onStatus?.(true)
    }
    ws.onmessage = (raw) => this.handleMessage(raw.data)
    ws.onclose = () => {
      this.onStatus?.(false)
      if (this.ws === ws) this.scheduleReconnect()
    }
    ws.onerror = () => {
      ws.close()
    }
  }

  private handleMessage(raw: unknown): void {
    let msg: {
      type?: string
      ts?: number
      data?: Record<string, unknown>
      level?: string
      source?: string
      code?: string
      message?: string
    }
    try {
      msg = JSON.parse(String(raw))
    } catch {
      return
    }
    if (!msg.type || !msg.data) return
    this.onEvent?.({
      type: msg.type,
      ts: typeof msg.ts === "number" ? msg.ts : Date.now() / 1000,
      data: msg.data,
      ...(msg.level !== undefined && { level: msg.level }),
      ...(msg.source !== undefined && { source: msg.source }),
      ...(msg.code !== undefined && { code: msg.code }),
      ...(msg.message !== undefined && { message: msg.message })
    })
    if (!WS_TYPES.has(msg.type)) return
    const d = msg.data
    const ts = typeof msg.ts === "number" ? msg.ts : Date.now() / 1000
    try {
      if (msg.type === "trade") {
        this.store.insertTrade({
          ts,
          pair: String(d.pair ?? ""),
          direction: Number(d.direction ?? 0),
          size: Number(d.size ?? 0),
          entry_price: Number(d.entry_price ?? 0),
          exit_price: Number(d.exit_price ?? 0),
          pnl: Number(d.pnl ?? 0),
          slippage_pips: Number(d.slippage_pips ?? 0),
          hold_steps: Number(d.hold_steps ?? 0),
          exit_reason: String(d.exit_reason ?? "")
        })
      } else if (msg.type === "order") {
        this.store.insertOrder({
          ts,
          pair: String(d.pair ?? ""),
          direction: Number(d.direction ?? 0),
          size: Number(d.size ?? 0),
          order_type: String(d.order_type ?? ""),
          status: String(d.status ?? ""),
          fill_price: d.fill_price == null ? null : Number(d.fill_price),
          slippage_pips: d.slippage_pips == null ? null : Number(d.slippage_pips),
          latency_us: d.latency_us == null ? null : Number(d.latency_us)
        })
      } else if (msg.type === "alert") {
        this.store.insertAlert({
          ts,
          level: String(msg.level ?? d.level ?? "info"),
          source: String(msg.source ?? d.source ?? "engine"),
          code: String(msg.code ?? d.code ?? ""),
          message: String(msg.message ?? d.message ?? "")
        })
      }
    } catch (e) {
      console.error("[audit] insert failed:", e)
    }
  }

  private scheduleReconnect(): void {
    if (this.stopped) return
    this.timer = setTimeout(() => {
      this.backoff = Math.min(this.backoff * 2, 30000)
      this.connect()
    }, this.reconnectDelay)
  }

  private closeSocket(): void {
    if (this.ws) {
      const w = this.ws
      this.ws = null
      try {
        w.onopen = null
        w.onmessage = null
        w.onclose = null
        w.onerror = null
        w.close()
      } catch {
        /* ignore */
      }
    }
  }
}
