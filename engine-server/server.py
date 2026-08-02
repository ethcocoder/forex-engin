"""FOREX DESK engine service — FastAPI REST + WebSocket broadcaster.

The engine owns all maths and workflow. This service exposes the engine to the
desktop over a JSON contract: REST for controls/config, WebSocket for the live
event stream. The desktop never imports or spawns the engine.

Event schema everywhere: ``{type, ts, data}`` (alerts additionally carry
``level`` / ``source`` / ``code`` / ``message`` at the top level).
"""

import asyncio
import json
import os
import queue
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

import config as cfg
import dataprep
import reports
from engine import SimulationEngine
from state import StateStore

store = StateStore()

# The engine thread (and data-prep thread) push events into this queue; a single
# asyncio task drains it and fans out to all connected WebSocket clients.
broadcast_queue: queue.Queue = queue.Queue()
clients: set = set()
clients_lock = asyncio.Lock()

sim = SimulationEngine(state=store, sink=lambda event: broadcast_queue.put_nowait(event))

dataprep.configure(store, lambda event: broadcast_queue.put_nowait(event))


def _json_default(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


async def broadcast(event: Dict[str, Any]) -> None:
    """Fan out one event to every connected WebSocket client (no per-client state)."""
    if not clients:
        return
    text = json.dumps(event, default=_json_default)
    dead: list = []
    async with clients_lock:
        for ws in list(clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)


async def _drain_loop() -> None:
    loop = asyncio.get_running_loop()
    while True:
        try:
            event = await asyncio.to_thread(broadcast_queue.get, True, 0.5)
        except queue.Empty:
            continue
        except Exception:
            return
        await broadcast(event)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Persisted readiness: data files on disk outlive a server restart, so
    # derive data_ready from disk (previously it reset to False on every boot).
    store.set_data_ready(dataprep.data_ready())
    task = asyncio.create_task(_drain_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="FOREX DESK Engine Service", version="0.1.0", lifespan=lifespan)

# Local loopback service — the Electron renderer connects directly over WS/fetch.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "FOREX DESK Engine Service", "status": "ok", "version": "0.1.0"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "sim_status": store.status,
        "data_ready": store.data_ready,
        "models_loaded": store.models_loaded,
        "data_preparing": dataprep.is_running(),
    }


# -- config ---------------------------------------------------------------
@app.get("/api/config")
def get_config():
    return cfg.get_config()


@app.put("/api/config")
def put_config(body: Dict[str, Any] = Body(default={})):
    try:
        return cfg.update_config(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# -- simulation ------------------------------------------------------------
@app.post("/api/sim/start")
def sim_start(body: Dict[str, Any] = Body(default={})):
    """Start a simulation. Optional body = transient overrides on the effective
    config (not persisted — use PUT /api/config for persistent settings)."""
    effective = cfg.merge_dicts(cfg.get_config(), body)
    return sim.start(effective)


@app.post("/api/sim/stop")
def sim_stop():
    return sim.stop()


@app.get("/api/sim/status")
def sim_status():
    return sim.status()


# -- history ---------------------------------------------------------------
@app.get("/api/trades")
def get_trades(limit: Optional[int] = Query(default=None, ge=1)):
    return {"trades": store.history("trades", limit)}


@app.get("/api/signals")
def get_signals(limit: Optional[int] = Query(default=None, ge=1)):
    return {"signals": store.history("signals", limit)}


@app.get("/api/equity")
def get_equity(limit: Optional[int] = Query(default=None, ge=1)):
    return {"equity": store.history("equity", limit)}


# -- reports ---------------------------------------------------------------
@app.get("/api/reports/tear-sheet")
def tear_sheet():
    return reports.tear_sheet(store)


@app.get("/api/reports/export")
def export(format: str = Query(default="csv", pattern="^(csv|json)$")):
    filename, content = reports.export(format, store)
    media_type = "text/csv" if format == "csv" else "application/json"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# -- data prep --------------------------------------------------------------
@app.post("/api/data/prepare")
def data_prepare():
    return dataprep.prepare()


# -- websocket ---------------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "snapshot", "ts": time.time(), "data": store.snapshot()})
    async with clients_lock:
        clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive / ping
    except WebSocketDisconnect:
        pass
    finally:
        async with clients_lock:
            clients.discard(ws)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("ENGINE_HOST", "127.0.0.1"),
        port=int(os.getenv("ENGINE_PORT", "8737")),
    )
