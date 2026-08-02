import { app, BrowserWindow, screen } from "electron"
import { readFileSync, mkdirSync, writeFileSync } from "fs"
import { join } from "path"

export interface WindowState {
  x?: number
  y?: number
  width?: number
  height?: number
  isMaximized?: boolean
}

const KEY = "windowState"

function settingsFile(): string {
  return join(app.getPath("userData"), "settings.json")
}

export function loadWindowState(): WindowState {
  try {
    const raw = JSON.parse(readFileSync(settingsFile(), "utf-8"))
    return (raw[KEY] as WindowState) ?? {}
  } catch {
    return {}
  }
}

export function saveWindowState(win: BrowserWindow): void {
  if (win.isDestroyed()) return
  try {
    const bounds = win.getNormalBounds()
    const state: WindowState = {
      x: bounds.x,
      y: bounds.y,
      width: bounds.width,
      height: bounds.height,
      isMaximized: win.isMaximized()
    }
    let existing: Record<string, unknown> = {}
    try {
      existing = JSON.parse(readFileSync(settingsFile(), "utf-8"))
    } catch {
      /* ignore */
    }
    mkdirSync(app.getPath("userData"), { recursive: true })
    writeFileSync(settingsFile(), JSON.stringify({ ...existing, [KEY]: state }, null, 2))
  } catch {
    /* ignore */
  }
}

export function applyWindowState(win: BrowserWindow, state: WindowState): void {
  const w = state.width ?? 1440
  const h = state.height ?? 900
  const x = state.x ?? 0
  const y = state.y ?? 0
  const visible = screen.getAllDisplays().some((d) => {
    const b = d.workArea
    return x >= b.x - 40 && x <= b.x + b.width - 40 && y >= b.y - 40 && y <= b.y + b.height - 40
  })
  win.setBounds(visible ? { x, y, width: w, height: h } : { x: 0, y: 0, width: w, height: h })
  if (state.isMaximized) win.maximize()
}
