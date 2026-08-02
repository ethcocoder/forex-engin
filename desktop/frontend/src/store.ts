import type { EngineEvent } from "../electron/ipc"

type Listener = (event: EngineEvent) => void

const TRANSIENT = new Set(["engine.status", "audit.connection"])

class Store {
  private listeners = new Map<string, Set<Listener>>()
  private history = new Map<string, EngineEvent[]>()
  private cap = 500

  subscribe(type: string, cb: Listener): () => void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set())
    this.listeners.get(type)!.add(cb)
    return () => this.listeners.get(type)?.delete(cb)
  }

  emit(event: EngineEvent): void {
    if (!TRANSIENT.has(event.type)) {
      const arr = this.history.get(event.type) ?? []
      arr.unshift(event)
      if (arr.length > this.cap) arr.length = this.cap
      this.history.set(event.type, arr)
    }
    this.listeners.get(event.type)?.forEach((cb) => cb(event))
  }

  events(type: string): EngineEvent[] {
    return this.history.get(type) ?? []
  }

  latest(type: string): EngineEvent | undefined {
    return this.history.get(type)?.[0]
  }
}

export const store = new Store()
