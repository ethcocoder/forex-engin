import { app, BrowserWindow, ipcMain, Menu, Tray, dialog } from "electron"
import { join, dirname } from "path"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs"
import { spawn, type ChildProcess } from "child_process"
import { AuditStore, AuditRecorder } from "./audit"
import { log } from "./log"
import { loadWindowState, saveWindowState, applyWindowState } from "./windowState"
import { Updater } from "./updater"
import type { EngineEvent, EngineHealth, StartResult, TearSheet } from "./ipc"

const DEFAULT_ENGINE_URL = "http://127.0.0.1:8737"

const settingsPath = (): string => join(app.getPath("userData"), "settings.json")

interface AppSettings {
  engineUrl?: string
  onboarded?: boolean
  minimizeToTray?: boolean
  updates?: { enabled?: boolean; feedUrl?: string }
}

function readSettings(): AppSettings {
  try {
    return JSON.parse(readFileSync(settingsPath(), "utf-8"))
  } catch {
    return {}
  }
}

function writeSettings(partial: Record<string, unknown>): void {
  const dir = app.getPath("userData")
  mkdirSync(dir, { recursive: true })
  writeFileSync(settingsPath(), JSON.stringify({ ...readSettings(), ...partial }, null, 2))
}

let engineUrl = DEFAULT_ENGINE_URL
let mainWindow: BrowserWindow | null = null
let auditStore: AuditStore | null = null
let auditRecorder: AuditRecorder | null = null
let tray: Tray | null = null
let simState = "idle"
let quitting = false
let engineProcess: ChildProcess | null = null
const updater = new Updater()

function findEnginePath(): string | null {
  const candidates = [
    join(dirname(app.getPath("exe")), "engine-server", "server.py"),
    join(app.getPath("userData"), "..", "forex-engin", "engine-server", "server.py"),
    "/home/nexuss0781/Desktop/Nex/forex-engin/engine-server/server.py"
  ]
  for (const p of candidates) {
    if (existsSync(p)) return p
  }
  return null
}

function startEngine(): void {
  if (engineProcess) return
  const serverPy = findEnginePath()
  if (!serverPy) {
    log("engine server.py not found", "warn")
    return
  }
  const serverDir = dirname(serverPy)
  const pythonBin = join(serverDir, ".venv", "bin", "python")
  const python = existsSync(pythonBin) ? pythonBin : "python3"
  log(`starting engine: ${python} ${serverPy} in ${serverDir}`)
  engineProcess = spawn(python, [serverPy], {
    cwd: serverDir,
    stdio: ["ignore", "pipe", "pipe"]
  })
  engineProcess.stdout?.on("data", (d: Buffer) => {
    const msg = d.toString().trim()
    if (msg) log(`engine: ${msg}`)
  })
  engineProcess.stderr?.on("data", (d: Buffer) => {
    const msg = d.toString().trim()
    if (msg) log(`engine: ${msg}`)
  })
  engineProcess.on("exit", (code) => {
    log(`engine exited with code ${code}`)
    engineProcess = null
  })
  engineProcess.on("error", (e) => {
    log(`engine spawn error: ${e.message}`, "error")
    engineProcess = null
  })
}

function stopEngine(): void {
  if (!engineProcess) return
  log("stopping engine")
  engineProcess.kill("SIGTERM")
  engineProcess = null
}

const wsUrlFor = (url: string): string => url.replace(/^http/, "ws") + "/ws"

async function fetchJson<T>(path: string, init?: RequestInit, timeoutMs = 30000): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${engineUrl}${path}`, { ...init, signal: controller.signal })
    if (!res.ok) throw new Error(`HTTP ${res.status} from ${path}`)
    return (await res.json()) as T
  } finally {
    clearTimeout(timer)
  }
}

function pushToRenderer(event: EngineEvent): void {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("engine:event", event)
  }
}

let lastHealthOk: boolean | null = null

async function pollHealth(): Promise<void> {
  try {
    const h = await fetchJson<EngineHealth>("/api/health", undefined, 10000)
    if (lastHealthOk !== true) log(`engine connected (sim=${h.sim_status}, data_ready=${h.data_ready})`)
    lastHealthOk = true
    pushToRenderer({
      type: "engine.status",
      ts: Date.now() / 1000,
      data: { state: "connected", ...h } as unknown as Record<string, unknown>
    })
  } catch {
    if (lastHealthOk !== false) log("engine unreachable", "warn")
    lastHealthOk = false
    pushToRenderer({
      type: "engine.status",
      ts: Date.now() / 1000,
      data: { state: "disconnected" }
    })
  }
}

function connectAuditRecorder(): void {
  if (!auditStore) return
  auditRecorder?.stop()
  auditRecorder = new AuditRecorder(
    wsUrlFor(engineUrl),
    auditStore,
    (connected) => {
      pushToRenderer({
        type: "audit.connection",
        ts: Date.now() / 1000,
        data: { connected }
      })
    },
    (event) => {
      if (event.type === "status" || event.type === "sim.status") {
        simState = String((event.data as Record<string, unknown>).status ?? simState)
        updateTray()
      }
      pushToRenderer(event as unknown as EngineEvent)
    }
  )
  auditRecorder.start()
}

function showMainWindow(): void {
  if (!mainWindow || mainWindow.isDestroyed()) return
  if (mainWindow.isMinimized()) mainWindow.restore()
  mainWindow.show()
  mainWindow.focus()
}

async function startSimFromTray(): Promise<void> {
  try {
    await fetchJson<StartResult>("/api/sim/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}"
    })
    log("sim start requested (tray/menu)")
    showMainWindow()
  } catch (e) {
    log(`sim start failed: ${e instanceof Error ? e.message : String(e)}`, "error")
  }
}

async function stopSimFromTray(): Promise<void> {
  try {
    await fetchJson<{ ok: boolean }>("/api/sim/stop", { method: "POST" })
    log("sim stop requested (tray/menu)")
  } catch (e) {
    log(`sim stop failed: ${e instanceof Error ? e.message : String(e)}`, "error")
  }
}

function updateTray(): void {
  if (!tray || !mainWindow || mainWindow.isDestroyed()) return
  tray.setToolTip(`FOREX DESK — ${simState.toUpperCase()}`)
}

function showAbout(): void {
  const counts = auditStore?.counts() ?? { trades: 0, orders: 0, alerts: 0 }
  const opts = {
    type: "info" as const,
    title: "About FOREX DESK",
    message: `FOREX DESK v${app.getVersion()}`,
    detail: `Engine URL: ${engineUrl}\nEngine: ${lastHealthOk === false ? "offline" : "online"}\nAudit: ${counts.trades} trades · ${counts.orders} orders · ${counts.alerts} alerts\n\nPure Node.js desktop client. The engine service runs separately.`
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    void dialog.showMessageBox(mainWindow, opts).catch(() => undefined)
  } else {
    void dialog.showMessageBox(opts).catch(() => undefined)
  }
}

function buildMenu(): void {
  Menu.setApplicationMenu(null)
}

function setupTray(): void {
  try {
    tray = new Tray(join(__dirname, "../../resources/icon.png"))
    tray.setToolTip(`FOREX DESK — ${simState.toUpperCase()}`)
    tray.setContextMenu(
      Menu.buildFromTemplate([
        { label: "Show", click: () => showMainWindow() },
        { type: "separator" },
        { label: "Run Simulation", click: () => void startSimFromTray() },
        { label: "Stop", click: () => void stopSimFromTray() },
        { type: "separator" },
        { label: "Quit", click: () => { quitting = true; app.quit() } }
      ])
    )
    tray.on("click", () => showMainWindow())
  } catch (e) {
    log(`tray init failed: ${e instanceof Error ? e.message : String(e)}`, "error")
  }
}

function registerIpc(): void {
  ipcMain.handle("config:get", () => fetchJson<Record<string, unknown>>("/api/config"))
  ipcMain.handle("config:set", (_e, partial: Record<string, unknown>) =>
    fetchJson<Record<string, unknown>>("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(partial)
    })
  )

  ipcMain.handle("sim:start", (_e, opts?: Record<string, unknown>) => {
    log(`sim:start called${opts ? ` opts=${JSON.stringify(opts)}` : ""}`)
    return fetchJson<StartResult>("/api/sim/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts ?? {})
    })
  })
  ipcMain.handle("sim:stop", () => {
    log("sim:stop called")
    return fetchJson<{ ok: boolean }>("/api/sim/stop", { method: "POST" })
  })
  ipcMain.handle("sim:status", () => fetchJson("/api/sim/status"))

  ipcMain.handle("reports:tear-sheet", () => fetchJson<TearSheet>("/api/reports/tear-sheet"))
  ipcMain.handle("reports:export", async (_e, format: "csv" | "json") => {
    const res = await fetch(`${engineUrl}/api/reports/export?format=${format}`)
    if (!res.ok) throw new Error(`HTTP ${res.status} from export`)
    const filename =
      format === "csv" ? "forexdesk_trades.csv" : "forexdesk_export.json"
    if (format === "csv") {
      return { filename, content: await res.text() }
    }
    return { filename, content: JSON.stringify(await res.json(), null, 2) }
  })

  ipcMain.handle("audit:trades", (_e, limit?: number) => auditStore?.recentTrades(limit) ?? [])
  ipcMain.handle("audit:orders", (_e, limit?: number) => auditStore?.recentOrders(limit) ?? [])
  ipcMain.handle("audit:alerts", (_e, limit?: number) => auditStore?.recentAlerts(limit) ?? [])

  ipcMain.handle("data:prepare", () => fetchJson<{ started: boolean }>("/api/data/prepare", { method: "POST" }))

  ipcMain.handle("engine:health", () => fetchJson<EngineHealth>("/api/health"))
  ipcMain.handle("engine:equity", (_e, limit?: number) =>
    fetchJson<{ equity: { ts: number; equity: number }[] }>(`/api/equity${limit ? `?limit=${limit}` : ""}`)
  )
  ipcMain.handle("engine:detect-binary", () => {
    const candidates: string[] = []
    const env = process.env["FOREXDESK_ENGINE_BIN"]
    if (env) candidates.push(env)
    const exeDir = dirname(app.getPath("exe"))
    candidates.push(join(exeDir, "forexdesk-engine"), join(exeDir, "bin", "forexdesk-engine"))
    for (const p of candidates) {
      try {
        if (existsSync(p)) return { found: true, path: p }
      } catch {
        /* ignore */
      }
    }
    return { found: false }
  })

  ipcMain.handle("app:get-onboarded", () => readSettings().onboarded === true)
  ipcMain.handle("app:set-onboarded", (_e, value: boolean) => {
    writeSettings({ onboarded: value === true })
    log(`onboarding ${value ? "completed" : "reset"}`)
    return { ok: true }
  })
  ipcMain.handle("app:get-minimize-tray", () => readSettings().minimizeToTray === true)
  ipcMain.handle("app:set-minimize-tray", (_e, value: boolean) => {
    writeSettings({ minimizeToTray: value === true })
    log(`minimize-to-tray ${value ? "enabled" : "disabled"}`)
    return { ok: true }
  })
  ipcMain.handle("app:get-updates", () => {
    const u = readSettings().updates
    return { enabled: u?.enabled === true, feedUrl: u?.feedUrl ?? "" }
  })
  ipcMain.handle("app:set-updates", (_e, value: { enabled: boolean; feedUrl: string }) => {
    const feedUrl = String(value.feedUrl ?? "").trim()
    writeSettings({ updates: { enabled: value.enabled === true, feedUrl } })
    log(`updates ${value.enabled ? "enabled" : "disabled"}${feedUrl ? ` (${feedUrl})` : ""}`)
    return { ok: true }
  })
  ipcMain.handle("engine:get-url", () => engineUrl)
  ipcMain.handle("engine:set-url", (_e, url: string) => {
    const clean = String(url).replace(/\/+$/, "")
    engineUrl = clean
    writeSettings({ engineUrl: clean })
    connectAuditRecorder()
    void pollHealth()
    return { ok: true, url: clean }
  })
}

function createWindow(): void {
  const winState = loadWindowState()
  mainWindow = new BrowserWindow({
    width: winState.width ?? 1440,
    height: winState.height ?? 900,
    backgroundColor: "#0B0F14",
    title: "FOREX DESK",
    show: false,
    webPreferences: {
      preload: join(__dirname, "../preload/preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  })
  applyWindowState(mainWindow, winState)

  mainWindow.once("ready-to-show", () => mainWindow?.show())

  if (process.env["ELECTRON_RENDERER_URL"]) {
    void mainWindow.loadURL(process.env["ELECTRON_RENDERER_URL"])
  } else {
    void mainWindow.loadFile(join(__dirname, "../renderer/index.html"))
  }

  mainWindow.webContents.setWindowOpenHandler(() => ({ action: "deny" }))
  mainWindow.on("close", (e) => {
    saveWindowState(mainWindow!)
    if (!quitting) {
      // minimize-to-tray: hide instead of closing unless the OS asks to quit
      const s = readSettings()
      if (s.minimizeToTray === true) {
        e.preventDefault()
        mainWindow?.hide()
        return
      }
    }
    mainWindow = null
  })
  const debouncedSave = (): void => saveWindowState(mainWindow!)
  mainWindow.on("resize", debouncedSave)
  mainWindow.on("move", debouncedSave)
  mainWindow.on("maximize", debouncedSave)
  mainWindow.on("unmaximize", debouncedSave)
  mainWindow.on("closed", () => {
    mainWindow = null
  })
  mainWindow.webContents.on("render-process-gone", (_e, details) => {
    log(`renderer gone: ${details.reason}`, "error")
  })
}

app.whenReady().then(() => {
  engineUrl = process.env["ENGINE_URL"] ?? readSettings().engineUrl ?? DEFAULT_ENGINE_URL

  const up = readSettings().updates
  updater.apply({ enabled: up?.enabled === true, feedUrl: up?.feedUrl })

  auditStore = new AuditStore(join(app.getPath("userData"), "audit.db"))
  connectAuditRecorder()

  registerIpc()
  Menu.setApplicationMenu(null)
  setupTray()
  createWindow()

  startEngine()

  void pollHealth()
  setInterval(() => void pollHealth(), 5000)

  setInterval(() => {
    auditStore?.prune()
    auditStore?.checkpoint()
  }, 10 * 60 * 1000)

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on("before-quit", () => {
  log("app quitting")
  stopEngine()
  auditStore?.prune()
  auditStore?.checkpoint()
  auditRecorder?.stop()
})

app.on("window-all-closed", () => {
  if (process.platform !== "darwin" || quitting) app.quit()
})

app.on("will-quit", () => {
  stopEngine()
  auditRecorder?.stop()
  auditStore?.close()
})
