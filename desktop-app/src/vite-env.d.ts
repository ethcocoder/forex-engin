/// <reference types="vite/client" />

interface Window {
  electronAPI?: {
    getStatus: () => Promise<{
      connected: boolean
      mode: string
      version: string
      heartbeat: string
      brokerConfig?: {
        brokerId: string
        apiKey: string
        accountId: string
        leverage: number
        mode: string
      }
      message: string
    }>
    saveConfig: (cfg: {
      brokerId: string
      apiKey: string
      accountId: string
      leverage: number
      mode: string
    }) => Promise<{ success: boolean; config: any }>
    runChaos: () => Promise<{ ok: boolean; output: string }>
  }
}
