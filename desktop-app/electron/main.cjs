const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('node:path')
const { execFile } = require('node:child_process')
const fs = require('node:fs')

let mainWindow

const configStorePath = path.join(app.getPath('userData'), 'broker_config.json')

function loadBrokerConfig() {
  try {
    if (fs.existsSync(configStorePath)) {
      return JSON.parse(fs.readFileSync(configStorePath, 'utf8'))
    }
  } catch {
    // fallback
  }
  return { brokerId: 'oanda_demo', apiKey: '', accountId: '', leverage: 10, mode: 'PAPER' }
}

function saveBrokerConfig(cfg) {
  try {
    fs.writeFileSync(configStorePath, JSON.stringify(cfg, null, 2), 'utf8')
    return true
  } catch {
    return false
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: '#070b14',
    title: 'Forex Engin · Elite10x Command Center',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })

  if (!app.isPackaged) {
    mainWindow.loadURL('http://127.0.0.1:5173')
    mainWindow.webContents.openDevTools({ mode: 'detach' })
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }
}

function engineBinary() {
  return path.resolve(__dirname, '../../cpp_engine/build/elite10x_engine')
}

ipcMain.handle('engine:status', async () => {
  const cfg = loadBrokerConfig()
  const hasCreds = Boolean(cfg.apiKey && cfg.accountId)
  return {
    connected: hasCreds,
    mode: cfg.mode || 'PAPER',
    version: 'elite10x-pr / production-desktop',
    heartbeat: new Date().toISOString(),
    brokerConfig: cfg,
    message: hasCreds ? `Connected to ${cfg.brokerId} (${cfg.mode} mode)` : 'Broker credentials are required for live data feeds. Currently running in safe paper mode.',
  }
})

ipcMain.handle('engine:saveConfig', async (_, cfg) => {
  const success = saveBrokerConfig(cfg)
  return { success, config: cfg }
})

ipcMain.handle('engine:runChaos', async () => new Promise((resolve) => {
  execFile(engineBinary(), { timeout: 15000 }, (error, stdout, stderr) => {
    if (error) {
      resolve({ ok: false, output: stderr || error.message })
      return
    }
    resolve({ ok: true, output: stdout })
  })
}))

app.whenReady().then(() => {
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
