/// <reference types="vite/client" />

interface Window {
  electronAPI?: {
    getStatus: () => Promise<{
      connected: boolean
      mode: string
      version: string
      heartbeat: string
      message: string
    }>
    runChaos: () => Promise<{ ok: boolean; output: string }>
  }
}
