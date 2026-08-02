# FOREX DESK — Implementation Phases

## How to use this document

This document is **self-contained**. It does not require you to read the engine
codebase. Each phase lists concrete tasks, expected files, and a "done when"
checklist. Follow phases in order.

There are **two independent deliverables**:

- **`engine-server/`** — a standalone Python FastAPI service that wraps the
  existing engine as a black box. Runs in its own process. Never ships inside
  the desktop.
- **`desktop/`** — the Electron app in pure Node.js. Never embeds Python. Talks
  to the engine **only through its API** (REST + WebSocket).

The API contract is the only seam between them. See `Design.md` §3 for the full
contract (it is identical regardless of where the engine runs).

---

## Phase 0 — Scaffold

**Goal:** Two clean, independent trees and a minimal engine service that boots
and answers `/api/health`.

### Tasks

1. Create the folder tree:

```
engine-server/
├── server.py
├── engine.py
├── state.py
├── reports.py
├── dataprep.py
└── requirements.txt

desktop/
├── frontend/
│   ├── electron/
│   └── src/
└── package.json
```

2. `engine-server/requirements.txt`:

```
fastapi>=0.110
uvicorn[standard]>=0.29
onnxruntime>=1.17
numpy>=1.24
pandas>=2.0
pydantic>=2.0
pyyaml>=6.0
```

> Note: the engine's simulation path does **not** import torch. Do not add
> torch, stable-baselines3, or hmmlearn — they would bloat the service.

3. `engine-server/server.py` — a minimal FastAPI app:
   - `GET /api/health` → `{ "status": "ok", "version": "0.1.0" }`
   - a root route `GET /` → app name + status
   - binds `127.0.0.1:8737` (configurable via `ENGINE_HOST` / `ENGINE_PORT`)

4. `desktop/package.json` — empty Electron placeholder (name, version).

5. Add to `.gitignore`: `engine-server/.venv/`, `frontend/node_modules/`,
   `dist/`, `build/`.

6. A one-line `README.md` at the repo root noting the two processes and how to
   run them side by side in dev.

### Done when

- `python engine-server/server.py` starts; `/api/health` returns 200 on
  `http://127.0.0.1:8737`.
- Both trees exist; the desktop tree contains **no Python**.
- `git status` shows only intended new files.

---

## Phase 1 — Engine service

**Goal:** The engine service loads configuration, starts/stops a paper-trading
simulation, and streams structured events over WebSocket. It is fully
self-contained and knows nothing about the desktop.

### 1.1 Configuration service

Files: `engine-server/config.py`

- Load the engine YAML config via the existing loader (import
  `configs.loader.load_config`, run from the repo root).
- Provide:
  - `get_config() -> dict` — full config for API consumers
  - `update_config(partial: dict) -> dict` — merge + persist overrides to
    `engine-server/user_config.yaml`
  - Balance is stored here (`initial_capital`), default **10,000**.
- User overrides win over defaults. Persistence is plain YAML — no database.
- The engine service is the **source of truth** for engine-affecting config.

### 1.2 Simulation engine

Files: `engine-server/engine.py`

- Wrap the existing quantized paper-trading run (the script
  `scripts/run_quantized_paper_trading.py`) **without modifying it**.
- Steps at start:
  1. Check data readiness: `data/EUR_USD_ticks.csv` and
     `data/EUR_USD_features.csv` exist. If missing → return a clear
     `error: DATA_MISSING` with instructions (link to "Prepare Data").
  2. Build the engine objects exactly as the script does (regime ensemble,
     scalers, model wrappers, aggregator, risk engine, paper broker,
     execution engine, performance tracker, pipeline).
  3. Run the tick loop in a background thread.
- Emit events from inside the loop by hooking the pipeline:
  - after each `process_tick` → `account`, `equity`, `progress`
  - when a signal fires → `signal`
  - when an order executes → `order`
  - when a trade closes → `trade` (include `exit_reason`)
  - risk/system events → `alert`
- Stop: set a `stop_event`; the loop checks it each tick and exits cleanly.
  A stopped run produces a final `done` status and the tear sheet.

### 1.3 State store

Files: `engine-server/state.py`

- Thread-safe in-memory store:
  - `status` (`idle | running | done | error`), `progress`
  - `account`, `positions`
  - ring buffers (bounded, e.g. 5,000) for `signals`, `trades`, `orders`, `alerts`, `equity`
- Query helpers used by REST history endpoints.

### 1.4 WebSocket broadcaster

Files: `engine-server/server.py`

- `WS /ws` — a single live channel.
- Every event message has one shape:
  ```json
  { "type": "...", "ts": 123, "data": { } }
  ```
  Types: `account`, `signal`, `order`, `trade`, `alert`, `equity`, `progress`,
  `sim.status`.
- On connect: send the current snapshot (`account`, `positions`, latest
  events) so the UI restores instantly.
- Broadcast is fan-out only — no per-client state.

### 1.5 Data prep

Files: `engine-server/dataprep.py`

- `POST /api/data/prepare` runs, as child processes from the repo root:
  1. `scripts/download_data.py`
  2. `scripts/generate_features.py --input data/EUR_USD_ticks.csv --output data/EUR_USD_features.csv`
- Stream progress lines to the WS as `progress` events.
- Report readiness via `/api/health.data_ready`.

### 1.6 REST surface

All from §3.1 of `Design.md`:

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | + `data_ready`, `models_loaded` |
| GET/PUT | `/api/config` | settings incl. balance |
| POST | `/api/sim/start` | body: optional config overrides |
| POST | `/api/sim/stop` | graceful stop |
| GET | `/api/sim/status` | status + progress + speed |
| GET | `/api/trades?limit=` | from state store |
| GET | `/api/signals?limit=` | from state store |
| GET | `/api/equity` | equity curve points |
| GET | `/api/reports/tear-sheet` | metrics + markdown |
| GET | `/api/reports/export?format=` | csv \| json |
| POST | `/api/data/prepare` | download + features |

### Done when

- Starting a sim with data present streams all 8 event types over WS.
- Stopping mid-run exits cleanly with `done`.
- Config PUT persists and survives service restart.
- `/api/data/prepare` produces the two CSVs and flips `data_ready`.
- The service is fully testable with zero desktop involvement (e.g. `curl`).

---

## Phase 2 — Electron shell

**Goal:** A native Node.js window opens, connects to the engine over HTTP/WS,
and records the audit. No Python anywhere in the desktop.

### 2.1 Main process

Files: `desktop/frontend/electron/main.ts`

- On app start:
  1. Read `ENGINE_URL` (default `http://127.0.0.1:8737`) from env / settings.
  2. Poll the engine `/api/health` until ready (timeout + failure UI).
  3. Create the BrowserWindow (width ~1440, height ~900, dark
     `backgroundColor: '#0B0F14'`).
- The desktop **never spawns or kills the engine** — it only connects.
- On quit: close the window; WS connections close on their own.

### 2.2 Audit store

Files: `desktop/frontend/electron/audit.ts`

- SQLite via `better-sqlite3`.
- Tables:
  - `trades (id, ts, pair, direction, size, entry, exit, pnl, slippage, hold_steps, exit_reason)`
  - `orders (id, ts, pair, direction, size, type, status, fill_price, slippage, latency_ns)`
  - `alerts (id, ts, level, source, code, message)`
- A WS client in main subscribes to `trade`/`order`/`alert` and inserts rows.
- IPC handlers expose `audit:trades`, `audit:orders`, `audit:alerts` to the
  renderer. History survives app restarts.

### 2.3 Preload

Files: `desktop/frontend/electron/preload.ts`

- `contextBridge.exposeInMainWorld('api', ...)`:
  - `config.get()`, `config.set()`
  - `sim.start()`, `sim.stop()`, `sim.status()`
  - `reports.tearSheet()`, `reports.export(format)`
  - `audit.trades()`, `audit.orders()`, `audit.alerts()`
  - `data.prepare()`
  - `engine.health()`, `engine.setUrl(url)`
  - `on(event, cb)` — WS subscription surface for the renderer

### 2.4 Scaffold

- `electron-vite` + React + TypeScript template.
- `src/` with the design tokens, a routing shell, and empty screen stubs.

### Done when

- Launching the app opens the window and connects to a running engine.
- A closed trade appears in SQLite and is queryable after app restart.
- The desktop runs with no Python runtime installed.

---

## Phase 3 — React UI

**Goal:** All 5 screens live, styled, and fed by the engine API.

### 3.1 Design tokens

Files: `desktop/frontend/src/styles/tokens.css`

Exactly the palette from `Design.md` §4:

- Background `#0B0F14`, surface `#121820`, border `#1E2833`
- Text `#F5F7FA` / muted `#8A94A6`
- Brand blue `#2E7CF6`, positive `#22C55E`, negative `#EF4444`, warning `#F59E0B`
- Fonts: Inter (UI), JetBrains Mono (numerals)
- Radius 4px, 1px hairlines, generous spacing

### 3.2 Shell

- Left sidebar (5 nav items, blue active state)
- Top status bar:
  - balance · equity · daily PnL · open positions · current regime
  - persistent `SIMULATION` badge
  - run / stop control

### 3.3 Screens

1. **Dashboard**
   - Equity curve chart (blue line, drawdown shading) — light custom canvas
     or a tiny chart lib; no heavy dependency
   - Metric cards: Sharpe · Max DD · Win Rate · Trades
   - Live signal ticker (direction, magnitude, confidence, regime)

2. **Signals**
   - Streaming table: time, direction, magnitude, confidence, regime,
     sub-model agreement

3. **Trades (Audit)**
   - Trades table: pair, direction, size, entry/exit, pnl (colored), slippage,
     exit reason, hold time; filters (by reason, by pair)
   - Order log below

4. **Reports**
   - Rendered tear sheet (markdown)
   - Export buttons: CSV, JSON (download via Electron)

5. **Settings**
   - Account: initial balance (number input, persisted via `PUT /api/config`)
   - Market: pair selection
   - Risk: kelly fraction, max risk %, circuit breakers
   - Engine: `ENGINE_URL` + connection status, broker (paper), God Mode toggle
   - Data: readiness indicator + "Prepare Data" button with progress

### 3.4 State handling

- Single WS client in the renderer (or via main + IPC).
- Simple store (React context or a ~50-line pub/sub). No heavy state lib.
- All numbers rendered in monospace; pnl colored; loading/empty/error states
  for every data view.
- If WS drops, show a calm "engine disconnected" banner and auto-reconnect.

### Done when

- Every screen updates live during a running simulation.
- Settings changes persist (via the engine) and take effect on the next run.
- The app looks calm, dark, and professional — blue + white only, plus
  semantic green/red/amber.

---

## Phase 4 — Packaging

**Goal:** An installable desktop app for Windows, Linux, macOS that connects to
an externally-running engine.

### 4.1 Desktop installer

- **electron-builder** in `desktop/package.json`:
  - Windows → NSIS
  - macOS → DMG
  - Linux → AppImage + deb
- The desktop package contains **only Node.js** — no Python, no torch, no
  engine files.

### 4.2 Engine deployment (separate)

- The engine service ships **independently** of the desktop:
  - dev / local: run from the repo (`python engine-server/server.py`).
  - optional: PyInstaller bundle (`forexdesk-engine`) for machines without a
    Python install.
- Data files are **not** bundled (downloaded via "Prepare Data").

### 4.3 Runtime wiring

- First run: desktop asks for `ENGINE_URL` (default localhost) and verifies
  `/api/health`. If unreachable, Settings shows the connection issue clearly.
- Bundled engine (if present) is a separate app the user starts — the desktop
  never launches it.

### Done when

- Each platform's installer builds and launches; UI connects to a running
  engine and runs a sim.
- Desktop install size stays small (no Python inside).

---

## Phase 5 — Polish

**Goal:** v1 quality — robust, honest, ready for the real gateway later.

### Tasks

- Empty-data onboarding: when `data_ready=false`, Settings shows a clear
  one-click path instead of errors.
- Error states: engine down, model load failure, sim crash — visible, calm,
  recoverable.
- Reports: pretty-print tear sheet, CSV/JSON exports verified.
- App icon, window title, about dialog.
- Audit export from the UI.
- Placeholder for the future real-money gateway: a disabled
  `LIVE MODE` toggle with a "coming soon" note, so the UI contract is ready.
- Performance pass: WS batching if event rate is high.

### Done when

- Fresh install → launch → prepare data → run sim → view dashboard →
  export report, all without touching a terminal.
- The `SIMULATION` badge and honest labeling are consistent everywhere.

---

## Cross-cutting rules

1. **Isolation is law.** The desktop never imports, spawns, or bundles the
   engine. API only.
2. **Simplicity first.** If a task needs a new library, ask whether a loop or
   a function suffices.
3. **No torch in the engine service.** Keep both sides small and portable.
4. **Engine math only in the engine service.** The renderer formats, never
   computes portfolio math.
5. **One event schema.** `{ type, ts, data }` everywhere — never invent a
   second shape.
6. **Commit per phase.** Each phase's "done when" block is the PR checklist.
