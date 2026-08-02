import { useEffect, useState } from "react"
import type { EngineEvent } from "../electron/ipc"
import { store } from "./store"

export function useLatest(type: string): EngineEvent | undefined {
  const [evt, setEvt] = useState<EngineEvent | undefined>(() => store.latest(type))
  useEffect(() => store.subscribe(type, setEvt), [type])
  return evt
}

export function useEvents(type: string): EngineEvent[] {
  const [evts, setEvts] = useState<EngineEvent[]>(() => store.events(type))
  useEffect(
    () => store.subscribe(type, () => setEvts([...store.events(type)])),
    [type]
  )
  return evts
}
