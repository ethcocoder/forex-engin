import { appendFileSync, mkdirSync } from "fs"
import { join } from "path"
import { app } from "electron"

let logPath: string | null = null

function ensurePath(): string {
  if (!logPath) {
    const dir = app.getPath("userData")
    mkdirSync(dir, { recursive: true })
    logPath = join(dir, "main.log")
  }
  return logPath
}

export function log(msg: string, level: "info" | "warn" | "error" = "info"): void {
  try {
    const line = `[${new Date().toISOString()}] [${level}] ${msg}\n`
    appendFileSync(ensurePath(), line)
  } catch {
    /* logging must never crash the app */
  }
}
