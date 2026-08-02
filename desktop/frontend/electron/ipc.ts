export interface EngineEvent {
  type: string
  ts: number
  data: Record<string, unknown>
  [key: string]: unknown
}

export interface EngineHealth {
  status: string
  version: string
  sim_status: string
  data_ready: boolean
  models_loaded: boolean
  data_preparing: boolean
}

export interface SimStatus {
  status: string
  error: string | null
  progress: {
    tick?: number
    total?: number
    speed_us?: number
    stopped_early?: boolean
  }
  running: boolean
}

export interface TearSheet {
  markdown: string
  metrics: Record<string, number>
}

export interface ExportResult {
  filename: string
  content: string
}

export interface StartResult {
  ok: boolean
  status?: string
  error?: string
}

export interface EngineApi {
  config: {
    get(): Promise<Record<string, unknown>>
    set(partial: Record<string, unknown>): Promise<Record<string, unknown>>
  }
  sim: {
    start(opts?: Record<string, unknown>): Promise<StartResult>
    stop(): Promise<{ ok: boolean }>
    status(): Promise<SimStatus>
  }
  reports: {
    tearSheet(): Promise<TearSheet>
    export(format: "csv" | "json"): Promise<ExportResult>
  }
  audit: {
    trades(limit?: number): Promise<unknown[]>
    orders(limit?: number): Promise<unknown[]>
    alerts(limit?: number): Promise<unknown[]>
  }
  data: {
    prepare(): Promise<{ started: boolean }>
  }
  engine: {
    health(): Promise<EngineHealth>
    getUrl(): Promise<string>
    setUrl(url: string): Promise<{ ok: boolean; url: string }>
    equity(limit?: number): Promise<{ equity: { ts: number; equity: number }[] }>
    detectBinary(): Promise<{ found: boolean; path?: string }>
  }
  app: {
    getOnboarded(): Promise<boolean>
    setOnboarded(value: boolean): Promise<{ ok: boolean }>
    getMinimizeTray(): Promise<boolean>
    setMinimizeTray(value: boolean): Promise<{ ok: boolean }>
    getUpdates(): Promise<{ enabled: boolean; feedUrl: string }>
    setUpdates(value: { enabled: boolean; feedUrl: string }): Promise<{ ok: boolean }>
  }
  on(type: string, cb: (event: EngineEvent) => void): () => void
}
