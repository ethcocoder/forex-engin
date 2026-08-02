import { contextBridge, ipcRenderer } from "electron"
import type { EngineApi, EngineEvent } from "./ipc"

const api: EngineApi = {
  config: {
    get: () => ipcRenderer.invoke("config:get"),
    set: (partial) => ipcRenderer.invoke("config:set", partial)
  },
  sim: {
    start: (opts) => ipcRenderer.invoke("sim:start", opts ?? {}),
    stop: () => ipcRenderer.invoke("sim:stop"),
    status: () => ipcRenderer.invoke("sim:status")
  },
  reports: {
    tearSheet: () => ipcRenderer.invoke("reports:tear-sheet"),
    export: (format) => ipcRenderer.invoke("reports:export", format)
  },
  audit: {
    trades: (limit) => ipcRenderer.invoke("audit:trades", limit),
    orders: (limit) => ipcRenderer.invoke("audit:orders", limit),
    alerts: (limit) => ipcRenderer.invoke("audit:alerts", limit)
  },
  data: {
    prepare: () => ipcRenderer.invoke("data:prepare")
  },
  engine: {
    health: () => ipcRenderer.invoke("engine:health"),
    getUrl: () => ipcRenderer.invoke("engine:get-url"),
    setUrl: (url) => ipcRenderer.invoke("engine:set-url", url),
    equity: (limit) => ipcRenderer.invoke("engine:equity", limit),
    detectBinary: () => ipcRenderer.invoke("engine:detect-binary")
  },
  app: {
    getOnboarded: () => ipcRenderer.invoke("app:get-onboarded"),
    setOnboarded: (value) => ipcRenderer.invoke("app:set-onboarded", value),
    getMinimizeTray: () => ipcRenderer.invoke("app:get-minimize-tray"),
    setMinimizeTray: (value) => ipcRenderer.invoke("app:set-minimize-tray", value),
    getUpdates: () => ipcRenderer.invoke("app:get-updates"),
    setUpdates: (value) => ipcRenderer.invoke("app:set-updates", value)
  },
  on: (type: string, cb: (event: EngineEvent) => void): (() => void) => {
    const listener = (_e: unknown, payload: EngineEvent): void => {
      if (type === "*" || payload.type === type) cb(payload)
    }
    ipcRenderer.on("engine:event", listener)
    return () => ipcRenderer.removeListener("engine:event", listener)
  }
}

contextBridge.exposeInMainWorld("api", api)
