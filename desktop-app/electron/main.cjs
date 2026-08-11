const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('node:path')
const { execFile } = require('node:child_process')

let mainWindow

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: '#070b14',
    title: 'Forex Engin · Command Center',
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

ipcMain.handle('engine:status', async () => ({
  connected: false,
  mode: 'PAPER',
  version: 'elite10x-pr / desktop-control-plane',
  heartbeat: new Date().toISOString(),
  message: 'Broker credentials are not configured. Paper mode is safe by default.',
}))

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
