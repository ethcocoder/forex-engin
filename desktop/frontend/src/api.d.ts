import type { EngineApi } from "../electron/ipc"

declare global {
  interface Window {
    api: EngineApi
  }
}

export {}
