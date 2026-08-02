# FOREX DESK — Detailed Work

Build log + implementation detail for every phase. Companion to `Design.md`
(what) and `Phase.md` (how). `todo.md` tracks per-task status; this file records
the concrete work done and any decisions made while building.

**Machine note:** the dev machine is slow (fastapi import ~10s). Always give
health-poll timeouts ≥ 30s and generous subprocess waits.

**Deployment note:** the engine runs on a high-resource cloud box; the desktop
app is a thin client that only talks REST + WebSocket over `ENGINE_URL`. The
dev machine's slow tick rate (~5 ticks/s) is irrelevant to correctness.

---

## Phase 0 — Scaffold (DONE)

- Created `engine-server/` stubs: `server.py`, `engine.py`, `state.py`,
  `reports.py`, `dataprep.py`, `requirements.txt`.
- Created `desktop/` tree: `package.json`, `README.md`, `frontend/electron/`,
  `frontend/src/`.
- Appended `.gitignore` (engine-server venv, node_modules, dist, build).
- `deps.sh`: creates venv, bootstraps pip via `ensurepip` (Ubuntu quirk —
  venv came without pip), installs deps with `.venv/bin/python -m pip`.
- Verified: `.venv/bin/python server.py` boots, `/api/health` returns
  `{"status":"ok","version":"0.1.0","data_ready":false,"models_loaded":false}`
  after ~8s.
- Installed deps: fastapi 0.141, uvicorn 0.52, onnxruntime 1.28, numpy 2.5,
  pandas 3.0, pydantic 2.13, pyyaml 6.0. **No torch.**

### Decisions
- Server boot is slow (~10s) → all health polling must tolerate ≥ 30s.
- deps.sh uses `python -m pip` (not `.venv/bin/pip`) because Ubuntu venvs
  lack a pip binary.

---

## Phase 1 — Engine service (IN PROGRESS)

### 1.1 Configuration service — `engine-server/config.py`

- `DEFAULT_CONFIG_PATH` = repo `configs/config.yaml`.
- `USER_CONFIG_PATH` = `engine-server/user_config.yaml`.
- `get_config() -> dict`:
  - load engine config via `configs.loader.load_config` (run from repo root).
  - deep-merge `user_config.yaml` on top → returns full effective config.
  - also inject `initial_capital` from `user_config` (default 10,000).
- `update_config(partial: dict) -> dict`:
  - deep-merge `partial` into `user_config.yaml`, write atomically.
  - returns the new effective config.
- Store `initial_capital` under `account.balance` in `user_config.yaml`.

### 1.2 State store — `engine-server/state.py`

Replace the stub with a thread-safe store:

- `self.status: str` — `idle | running | done | error`
- `self.error: str | None`
- `self.progress: dict` — `{tick, total, speed_us}`
- `self.account: dict` — cash/equity/pnl fields
- `self.positions: dict` — pair → {size, entry}
- ring buffers (`collections.deque`, maxlen=5000):
  - `signals`, `trades`, `orders`, `alerts`, `equity`
- methods: `snapshot() -> dict` (for WS connect), `history(name, limit)`,
  `reset()` (called before a new run).

### 1.3 Simulation engine — `engine-server/engine.py`

Replace the stub with `SimulationEngine`:

- `__init__()`: holds `stop_event`, `thread`, `state: StateStore`, `broker`,
  `tracker`, `pipeline`, `aggregator` (populated on start).
- `start(config) -> dict`:
  1. `state.reset()`, `state.status = "running"`.
  2. Data readiness check: `data/EUR_USD_ticks.csv` and
     `data/EUR_USD_features.csv`. Missing → `status = "error"`, return
     `{"error": "DATA_MISSING"}`.
  3. Build engine objects exactly like
     `scripts/run_quantized_paper_trading.py`:
     - `ONNXRegimeEnsembleEstimator`, regime scaler, feature scaler
     - `EnsembleAggregator` + wrappers (`temporal` FP32, `maml`/`rl` INT8,
       regime wrapper)
     - `RiskEngine` + `KellySizer` + `DrawdownFilter` + `SpreadFilter`
     - `PaperBroker`, `ExecutionEngine`, `PerformanceTracker`
     - God Mode components (synapse, mesh, attacker, kernel bypass) —
       wrapped in try/except, degrade gracefully.
  4. Launch the tick loop in a background thread.
- `stop()`: set `stop_event`, join thread (timeout), set `status = "done"`.
- `status() -> dict`: status, error, progress.
- Emits every event into `state` (state is the single source of truth;
  the WS layer broadcasts from it).

**Tick loop (thread):**
- iterate `range(seq_len-1, len(features))`.
- per tick: build `X_input` window, `market_data` dict, call
  `pipeline.process_tick(...)`.
- after each tick push:
  - `state.account` (from broker cash + tracker),
  - `state.positions` (broker.get_positions + entry prices),
  - `state.equity.append({ts, equity})`,
  - `state.progress = {tick, total, speed_us}`.
- hook into `pipeline` to capture `signal`, `order`, `trade`, `alert` events:
  - wrap/observe `ensemble.predict` → on direction != 0, push `signal`.
  - observe `execution_engine.execute` → push `order` (status/fill/slippage).
  - observe trade closes in `RealTimePipeline` (the exit block) → push `trade`
    with `exit_reason` (stop_loss / decay / drawdown / reversal).
  - watch `logger`-level risk events via the risk engine return metadata and
    pipeline branches → push `alert`.
- check `stop_event` each tick; break cleanly.
- on completion: final mark-to-market, `state.status = "done"`, generate
  tear sheet and store in `state.reports`.

**Implementation approach for hooks:** subclass/observe the existing pipeline
class in the engine module (a `BridgePipeline(RealTimePipeline)` override that
calls `self.on_*` callbacks), so the engine scripts stay untouched.

### 1.4 WebSocket broadcaster — `engine-server/server.py`

- `clients: set[WebSocket]` guarded by an `asyncio.Lock`.
- `async def ws_endpoint(ws)`: accept, add client, send `state.snapshot()`,
  loop receiving (keeps alive), remove on disconnect.
- `broadcast(event: dict)`: json-serialize and send to all clients.
- Bridge between the simulation thread and asyncio loop:
  - a `queue.Queue` that the engine thread pushes events into;
  - an asyncio task polls the queue and calls `broadcast`.
- Event shape everywhere: `{type, ts, data}`.

### 1.5 Data prep — `engine-server/dataprep.py`

- `prepare() -> dict` runs two subprocesses from the repo root:
  1. `.venv/bin/python scripts/download_data.py`
  2. `.venv/bin/python scripts/generate_features.py --input data/EUR_USD_ticks.csv --output data/EUR_USD_features.csv`
- Stream stdout/stderr lines into the event queue as
  `{type: "progress", data: {message}}`.
- On success set `data_ready` in health.
- Runs in its own thread so the API stays responsive.

### 1.6 REST surface — `engine-server/server.py`

All endpoints:

| Method | Path | Handler |
|---|---|---|
| GET | `/` | app info |
| GET | `/api/health` | status + `data_ready` + `models_loaded` |
| GET | `/api/config` | `config.get_config()` |
| PUT | `/api/config` | `config.update_config(body)` |
| POST | `/api/sim/start` | `engine.start(body)` (body = optional overrides) |
| POST | `/api/sim/stop` | `engine.stop()` |
| GET | `/api/sim/status` | `engine.status()` |
| GET | `/api/trades?limit=` | `state.history("trades", limit)` |
| GET | `/api/signals?limit=` | `state.history("signals", limit)` |
| GET | `/api/equity` | `state.history("equity")` |
| GET | `/api/reports/tear-sheet` | `reports.tear_sheet(state)` |
| GET | `/api/reports/export?format=` | csv \| json of trades/signals |
| POST | `/api/data/prepare` | `dataprep.prepare()` |

### 1.7 Reports — `engine-server/reports.py`

- `tear_sheet(state) -> dict`: parse the tracker's markdown tear sheet
  (or expose structured metrics) → `{markdown, metrics}`.
- `export(format) -> (filename, bytes)`:
  - csv: trades table as CSV
  - json: `{trades, signals, equity}` as JSON.

### Done when
- With data present, starting a sim streams all 8 event types over WS.
- Stop mid-run exits cleanly → `done`.
- `PUT /api/config` persists and survives service restart.
- `/api/data/prepare` produces the two CSVs and flips `data_ready`.
- Full service testable with `curl` — no desktop involved.

### Fixes found during live verification (2026-08-01)
- **Timezone:** `dataprep._convert_raw()` failed on mixed timezones
  (`Date`/`Datetime` columns). Fix: `pd.to_datetime(df["timestamp"], utc=True)`
  after renaming. Data-prep then produced `data/EUR_USD_ticks.csv` (12,348 bars)
  and `data/EUR_USD_features.csv` (57 features), `data_ready=true`.
- **CWD:** engine thread inherited the server's CWD (`engine-server/`), so the
  script's relative `saved_models/*` paths broke (`MODEL_MISSING`). Fix:
  `os.chdir(ENGINE_ROOT)` at the top of `SimulationEngine._run()` — mirrors
  `scripts/run_quantized_paper_trading.py`, which assumes repo-root CWD.
- **lightgbm:** unpickling the saved ensemble/aggregator pickles requires the
  `lightgbm` module (the pickle references `LGBMRegressor` classes; the
  aggregator's own Ridge fallback doesn't help). Added `lightgbm` to
  `requirements-sim.txt` (has prebuilt manylinux wheels).
- **Live sim (2026-08-01):** full loop verified on this machine — model load
  ~90s, then tick loop. WS streamed all event types: `alert` (God Mode
  adversarial vulnerability), `signal` (confidence/regime), `order`
  (FILLED/slippage), `trade` (pnl + exit_reason), `account`/`positions`/
  `equity`, `progress`, `status`.

---

## Phase 2 — Electron shell

**Goal:** a native Node-only window opens, connects to the engine over
HTTP/WS, and records the audit. No Python anywhere in `desktop/`.

**Starting state:** `desktop/` contains only `package.json` (placeholder),
`README.md`, `frontend/src/.gitkeep`, `frontend/electron/.gitkeep`. Engine
service (Phase 1) is DONE and verified; restart it for desktop work with:
`.venv/bin/python server.py` from `engine-server/` (see `Server PIDs` note below).

**Environment (this dev box):** Node v24.18.0, npm 11.16.0, live display on
`:0` (Cinnamon) so Electron can open a real window — no xvfb needed.

### Stack / dependencies (decided)
- **electron-vite** + React + TypeScript. Keep the plan's directory layout:
  `desktop/frontend/electron/{main,audit,preload}.ts`,
  `desktop/frontend/src/...`. electron-vite defaults to `src/main`,
  `src/preload`, `src/renderer` — **must override entry paths** in
  `electron.vite.config.ts` (see below).
- **WS client lives in the main process** (`ws` package): audit persistence
  runs even if the renderer is closed. Renderer subscribes via IPC
  (`engine:event` push + `on()`).
- **better-sqlite3** for the audit DB, rebuilt against Electron's ABI via
  `@electron/rebuild` postinstall. If the native rebuild fails, fall back to
  `node:sqlite` (experimental in Electron's bundled Node).
- **react-router-dom** for the shell; a ~50-line pub/sub store (no heavy state
  lib); no chart lib yet (Phase 3).

### 0. Toolchain setup (do first)
1. Write `desktop/package.json`:
   - scripts: `dev` (electron-vite dev), `build` (electron-vite build),
     `typecheck` (tsc --noEmit -p tsconfig.web.json + tsconfig.node.json),
     `rebuild` (@electron/rebuild -f -w better-sqlite3), `postinstall`
     (electron-rebuild).
   - `main`: `out/main/index.js` (electron-vite output).
   - devDeps: `electron`, `electron-vite`, `vite`, `@vitejs/plugin-react`,
     `typescript`, `@types/node`, `@types/react`, `@types/react-dom`,
     `@types/ws`, `@electron/rebuild`.
   - deps: `react`, `react-dom`, `react-router-dom`, `ws`,
     `better-sqlite3`.
2. `desktop/electron.vite.config.ts`:
   - `main.entry = { index: "frontend/electron/main.ts" }`
   - `preload.entry = { index: "frontend/electron/preload.ts" }`
   - `renderer.root = "frontend/src"` with an `index.html` inside it
     (electron-vite copies it to `out/renderer/index.html`).
3. TS configs: `tsconfig.node.json` (main/preload + electron.vite.config.ts,
   `module: ESNext`, `moduleResolution: bundler`, `types: ["node"]`) and
   `tsconfig.web.json` (renderer, jsx: react-jsx, dom libs). Root
   `tsconfig.json` references both.
4. `npm install` (Electron binary ~100MB download — needs network).
5. If `better-sqlite3` was installed without rebuild, run `npm run rebuild`.
   Sanity: `npx electron -e "require('better-sqlite3')(':',{})"` — a version
   mismatch throws ABI errors; fix with `electron-rebuild`.

### 2.1 Main process — `desktop/frontend/electron/main.ts`
- Read engine URL in priority: env `ENGINE_URL` → `userData/settings.json`
  (`{engineUrl}`) → default `http://127.0.0.1:8737`. Persist via IPC
  `engine:setUrl`.
- Health poll: fetch `{url}/api/health` with a 30s timeout (engine boot is
  slow). Retry loop while window is hidden/loading; expose status to the
  renderer via `engine:health` (IPC invoke) and push `engine:status` events on
  change (connected / connecting / failed + lastError).
- BrowserWindow 1440×900, `backgroundColor: "#0B0F14"`, `webPreferences: {
  preload, contextIsolation: true, nodeIntegration: false, sandbox: false }`.
- Create the window even if the engine is down (renderer shows the failure
  state) — do not block app start on the poll.
- **Never spawn/kill the engine.** On `window-all-closed` (non-darwin) `quit()`.
- Register all `ipcMain.handle` channels (names in §2.3); a `settings.ts`
  helper reads/writes `userData/settings.json`.

### 2.2 Audit store — `desktop/frontend/electron/audit.ts`
- `new Database(app.getPath("userData") + "/audit.db")`.
- Schema (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL indexed):
  - `trades(id, ts, pair, direction, size, entry_price, exit_price, pnl,
    slippage_pips, hold_steps, exit_reason)`
  - `orders(id, ts, pair, direction, size, order_type, status, fill_price,
    slippage_pips, latency_us)`
  - `alerts(id, ts, level, source, code, message)`
- WS client: `new WebSocket(wsUrl)` where `wsUrl = engineUrl.replace(/^http/,
  "ws") + "/ws"`. On message: parse `{type, ts, data}`; for types
  `trade`/`order`/`alert`, `INSERT` (prepared statements). Reconnect with
  backoff on drop/error. A new WS client is (re)built on engine URL change.
- IPC: `ipcMain.handle("audit:trades", ...)`, `audit:orders`, `audit:alerts`
  → `SELECT * FROM x ORDER BY ts DESC LIMIT ?`.
- Unit-testable standalone: export a class `AuditStore(path)` that works
  without Electron (test with `node` + better-sqlite3 under the system Node —
  note ABI: if system Node's better-sqlite3 differs from Electron's, test
  against the rebuilt binary via `npx electron` or keep tests logic-only).

### 2.3 Preload — `desktop/frontend/electron/preload.ts`
- `contextBridge.exposeInMainWorld("api", {...})`, all `invoke` wrappers:
  - `config.get()`, `config.set(partial)`
  - `sim.start(opts?)`, `sim.stop()`, `sim.status()`
  - `reports.tearSheet()`, `reports.export(format)` (returns buffer + filename
    for a `dialog.showSaveDialog` save)
  - `audit.trades(limit?)`, `audit.orders(limit?)`, `audit.alerts(limit?)`
  - `data.prepare()`
  - `engine.health()`, `engine.setUrl(url)`
  - `on(type, cb)` — subscribes to main→renderer pushes (`engine:status`,
    `engine:event`); returns an unsubscribe fn.
- Main pushes WS events to the focused window with
  `webContents.send("engine:event", event)` so the renderer never opens its
  own WS.

### 2.4 Renderer scaffold — `desktop/frontend/src/`- `index.html` (loads `/src/main.tsx`, CSP meta allowing connect-src to the
  engine origin), `main.tsx`, `App.tsx`.
- `styles/tokens.css` — exact palette from Design.md §4: bg `#0B0F14`, surface
  `#121820`, border `#1E2833`, text `#F5F7FA`/muted `#8A94A6`, brand `#2E7CF6`,
  pos `#22C55E`, neg `#EF4444`, warn `#F59E0B`; radius 4px; 1px hairlines;
  Inter + JetBrains Mono.
- `store/` — a small pub/sub store: subscribes via `window.api.on`,
  dispatches by event type to registered screens, exposes snapshot getters.
- `routes/` — RouterProvider with sidebar (5 nav items, blue active state),
  top status bar (balance · equity · daily PnL · positions · regime ·
  `SIMULATION` badge · run/stop), and stub screens `Dashboard`, `Signals`,
  `Trades`, `Reports`, `Settings` (each a typed component showing its data
  contract from Design.md §3.2–3.6).

### Verification (Phase 2 "done when")
1. `npm run typecheck` clean (both tsconfigs).
2. `npm run build` → `desktop/out/` has `main/index.js`, `preload/index.js`,
   `renderer/index.html`.
3. Restart engine server (`.venv/bin/python server.py` from `engine-server/`),
   then `npm run dev` — window opens on `:0`, connects, `/api/health` green.
4. Start a sim from the UI (or curl) → assert `audit.db` grows rows in
   `trades`/`orders`/`alerts`.
5. Quit + relaunch → audit rows still queryable (persistence proof).
6. File scan: `find desktop -name "*.py"` returns nothing.

### Server PIDs (dev helpers)
- Keep the engine server's pid in `/tmp/opencode/forex-server.pid` when
  launching: `nohup .venv/bin/python server.py > /tmp/opencode/forex-server.log
  2>&1 & echo $! > /tmp/opencode/forex-server.pid; disown`.
- Kill by pid, NOT `pkill -f "server.py"` (that pattern matches the invoking
  shell). If pid is lost: `fuser -k 8737/tcp`.
- Health-poll timeouts must be ≥ 30s (this machine imports fastapi in ~10s).

### Fixes found during Phase 2 build (2026-08-01)
- **npm v11 blocks postinstall scripts by default.** esbuild, electron, and
  electron-rebuild silently skip their install steps → run
  `npm approve-scripts --all` (and per-package as needed) after install, or
  build fails at runtime with missing binaries.
- **Vite peer pinning:** electron-vite 5 needs `vite@^7` (v8 conflicts) and
  `@vitejs/plugin-react@^5` (v6 needs vite 8). Do NOT install `vite@latest` /
  `@vitejs/plugin-react@latest`.
- **electron-vite output filenames derive from the entry filename**, not
  `index`. Entry `main.ts` → `out/main/main.js`, `preload.ts` →
  `out/preload/preload.js`. Set package.json `"main"` and the preload path in
  `webPreferences` accordingly.
- **Renderer root:** with `renderer.root = frontend/src`, the HTML script tag
  must be relative (`./main.tsx`), not `/src/main.tsx` (absolute resolves
  against root).
- **CSS side-effect imports need types:** add `frontend/src/vite-env.d.ts`
  with `/// <reference types="vite/client" />` for `import "../styles/app.css"`.
- **better-sqlite3 types:** `@types/better-sqlite3` is a devDependency.
- **`pkill -f` self-match trap is NOT limited to python:** any pattern that
  appears verbatim in the invoking shell (e.g. `pkill -f "electron/dist/electron"`)
  kills the shell. Use bracket patterns (`[e]lectron/...`) or kill by PID.
- **Kill Electron via the pid of the `.bin/electron` child, not the npm
  wrapper** (`nohup npx electron` → the pid file holds npm's pid; killing it
  orphans the Electron process → duplicate windows).
- **Black window on launch (two bugs):**
  1. `<HashRouter>` was in BOTH `main.tsx` (around `<App/>`) and `App.tsx` →
     React Router threw "cannot render a Router inside another Router" and
     nothing mounted. Keep the router in exactly one place (App.tsx).
  2. `tokens.css` was referenced as a raw `<link>` in `index.html`; Vite
     dropped it during build (not imported from JS). Result: styles missing
     at runtime → "black-ish" unstyled window. Fix: `import "./styles/tokens.css"`
     in `main.tsx` so it lands in the bundle. Verified via CDP:
     bodyBg `rgb(11,15,20)`, Inter, sidebar `rgb(18,24,32)`.

---

## Phase 3 — React UI

**Goal:** all 5 screens live, polished, and fed by the engine API — calm, dark,
professional, blue + white only. Every screen updates live during a sim.

**Starting state:** Phase 2 renderer already has the design tokens
(`tokens.css`), the app shell (`Layout.tsx`: sidebar + topbar with
balance/equity/PnL + SIMULATION badge + Run/Stop), a pub/sub store
(`store.ts`, ~500-event cap, wildcard `window.api.on("*")` subscription set up
in `main.tsx`), hooks (`useLatest`/`useEvents`), formatting helpers
(`format.ts`), and 5 *functional* screens (Dashboard metric cards + live
tables, Signals table, Trades audit+orders, Reports tear-sheet + export,
Settings engine/account/data). All of it already streams live events.

**Decisions (simplicity first):**
- **Equity chart = custom SVG component** (~120 lines), zero chart deps.
- **Tear-sheet markdown = custom mini-renderer** (~30 lines) targeting the
  known markdown structure (H1/H2, `**bold**`, `*` bullets, `---`). No md lib.
- **No new npm dependencies** unless a genuine need appears.

### Existing file map (renderer)
```
desktop/frontend/src/
├── main.tsx          # window.api.on("*") → store.emit; imports tokens.css
├── App.tsx           # HashRouter + Layout + Banner + 5 routes
├── store.ts          # pub/sub, history cap 500, excludes transient types
├── hooks.ts          # useLatest(type), useEvents(type)
├── format.ts         # dirArrow, dirClass, fmtNum, fmtPct, fmtTs, fmtHold
├── api.d.ts          # Window.api: EngineApi (from frontend/electron/ipc.ts)
├── components/Layout.tsx   # sidebar + topbar + Outlet + Run/Stop
├── screens/{Dashboard,Signals,Trades,Reports,Settings}.tsx
└── styles/{tokens.css,app.css}
```

### Data wiring reference (ground truth from the engine)
- Live events over WS (type → payload `data`): `account`, `positions`
  (pair→{size,entry}), `signal`, `order`, `trade`, `alert` (level/source/
  code/message top-level + `data`), `equity`, `progress` (tick/total/
  speed_us), `status`/`sim.status`.
- REST via `window.api` IPC: `engine.health()`, `sim.status()`,
  `sim.start(opts)`, `reports.tearSheet()` → `{markdown, metrics}`,
  `reports.export("csv"|"json")` → `{filename, content}`,
  `audit.trades/orders/alerts(limit)`, `config.get()/set(partial)`,
  `engine.getUrl()/setUrl(url)`, `data.prepare()`.
- Config keys the engine persists (`config.py` `USER_EDITABLE`): top-level
  `account` (→ `{balance}`), `pairs` (list), `risk` with sub-keys `sizing`
  (`method`, `kelly_fraction`, `max_account_risk_pct`), `circuit_breakers`
  (`daily/weekly/monthly_drawdown_limit`), `limits`, `monitoring`. Everything
  else is engine-owned and ignored on PUT. See `configs/config.yaml` for
  defaults (kelly 0.15, max risk 0.0075, circuit breakers 0.03/0.06/0.10).

### Task 3.1 — Equity chart `src/components/EquityChart.tsx`
- Props: `points: {ts:number; equity:number}[]` (+ optional height).
- Pure SVG: `<polyline>` in brand blue `#2E7CF6`, `fill` area with low-opacity
  blue, **drawdown shading** = fill below the running max under a semi-dark
  overlay in `#EF4444` at 8–10% opacity (Design.md: "drawdown shading").
- Auto-scale Y to `[min, max]` with ~5% padding; X spans index range.
- Labels (JetBrains Mono, `var(--mono)`): min, max, latest equity, plus
  drawdown % badge. `viewBox` responsive; thousands of points OK (polyline).
- Empty state: dashed placeholder + "No equity data yet — start a sim".
- Re-render: pure function of props; parent throttles (see Task 3.3).

### Task 3.2 — Top status bar upgrade `src/components/Layout.tsx`
- Existing: Balance, Equity, Daily PnL (mono numerals; pnl green/red),
  SIMULATION badge, Run/Stop (disabled by state).
- Add: **Open positions** (count + notional from latest `positions` event:
  count = Object.keys(data).length; notional = Σ size) and **Regime** (from
  latest `signal` event `data.regime` → label map {0..n} or raw int).
- Add **sim status pill**: idle/loading/running/done/error — from
  `sim.status()` on mount + `status`/`sim.status` events; color: running=blue,
  done=green, error=red, idle=muted.
- Keep the Banner logic (data_ready warning) in App.tsx; ensure it hides when
  health reports `data_ready: true` (recheck on `engine.status` change).

### Task 3.3 — Dashboard polish `src/screens/Dashboard.tsx`
- Metrics row: Equity, Daily PnL, Sharpe, Max DD, Win Rate, Trades —
  live from `account` events + tear-sheet metrics (fetch on mount; update when
  sim transitions to `done`).
- Equity panel: fetch full curve via `/api/equity` (through a new IPC call —
  see Task 3.8) on mount; merge with live `equity` events; render
  `EquityChart`. Throttle re-render to ~250ms (chart doesn't need 100%/s).
- Live signal ticker: last `signal` event → direction arrow (colored),
  magnitude, confidence, regime; "no signal yet" empty state.
- Recent trades table (last 8, live) + loading/empty/error states.

### Task 3.4 — Signals screen `src/screens/Signals.tsx`
- Streaming table: time, direction (colored arrow), magnitude, confidence,
  uncertainty, regime, sub-model agreement (each `sub_models` key=value).
- Filters (client-side, presentation only): regime dropdown + minimum
  confidence slider/input. Monospace numerals, scroll container.
- Loading/empty states; pause-on-hover not needed.

### Task 3.5 — Trades screen `src/screens/Trades.tsx`
- Merge sources: audit rows (`window.api.audit.trades(limit)`) + live `trade`
  events (dedupe by ts+pnl or prepend only if not already present).
- **Filters**: exit_reason (All/stop_loss/decay/drawdown/reversal) and pair
  (All + distinct pairs) dropdowns, applied client-side.
- Table: time, pair, direction, size, entry, exit, **colored pnl**,
  slippage, hold (bars), reason. Summary line: filtered count + Σ pnl.
- Order log below (live `order` events): time, dir, type, size, fill, status,
  slippage, latency µs.
- Empty/error states; audit fetch failure → fall back to live-only + note.

### Task 3.6 — Reports screen `src/screens/Reports.tsx`
- Mini-markdown renderer `src/components/Markdown.tsx`: split lines → H1/H2
  (`# `, `## `), `**bold**`, `* ` bullets, `---` → `<hr>`, else paragraph
  (monospace for numbers). Sanitize by construction (known shapes only; no
  raw HTML passthrough).
- Metrics grid (live from tear-sheet fetch + `done` update).
- Export CSV/JSON buttons → `window.api.reports.export(...)` →
  Blob download (already wired; keep).
- States: loading, error (engine down), empty (no run yet).

### Task 3.7 — Settings screen `src/screens/Settings.tsx` (all 5 sections)
- **Account**: initial balance (number) → `config.set({account:{balance}})`.
- **Market**: pairs multi-select (checkboxes) → `config.set({pairs:[...]})`
  using the pair list from `config.get()`; allow comma input too.
- **Risk**: kelly fraction, max account risk %, daily/weekly/monthly
  circuit-breaker limits (number inputs, percent) →
  `config.set({risk:{sizing:{...}, circuit_breakers:{...}}})`.
- **Engine**: ENGINE_URL + Connect (existing) + connection status + broker
  `paper` (read-only from config) + **God Mode toggle** — persists to
  `user_config` (via `config.set({simulation:{god_mode:bool}})` if the engine
  tolerates it; else store locally and pass as `sim.start({god_mode:...})`
  override). Label honestly: "applies on the next run".
- **Data**: readiness indicator + Prepare Data button (existing) + **progress
  bar** fed by live `progress` events (tick/total → %, plus the `progress`
  text messages streamed during prepare).
- Load current values on mount from `config.get()` + `engine.getUrl()` +
  `engine.health()`; save buttons per section with "Saved." feedback.

### Task 3.8 — IPC additions (main + preload)
- Add `engine:equity` handler in `main.ts` → `GET /api/equity` (returns the
  full curve array) and expose `engine.equity()` in `ipc.ts` +
  `preload.ts`. Dashboard uses it for the initial chart.
- (Optional, only if needed) `engine:config-schema` not needed — Settings
  reads `config.get()` directly.

### Task 3.9 — Connection resilience
- `App.tsx` Banner → dual banner: engine down ("Engine offline — start the
  engine service, then Settings → Engine URL") vs data not prepared.
- On `engine.status` events with `state: disconnected`, show a persistent
  "Reconnecting…" banner; auto-clears on `connected`. Main already reconnects
  the WS recorder with backoff; renderer just reflects status.
- All screens already have loading/empty/error states — sweep and standardize
  the error copy.

### Task 3.10 — Design pass `src/styles/{tokens.css,app.css}`
- Standardize: 4px radius, 1px hairlines, spacing scale (4/8/12/16/24),
  form inputs (dark surface-2, hairline, focus ring `--brand`), table row
  hover, scrollbar styling (thin, `--border`), `.pill` for status chips.
- Ensure semantic colors ONLY for meaning: pos/neg pnl, amber alerts.
- Consistency: every numerals cell uses `.num`; SIMULATION badge everywhere.

### Verification (Phase 3 "done when")
1. `npm run typecheck` + `npm run build` clean.
2. Engine + app running; start a sim from the UI → Dashboard chart fills and
   grows live; Signals/Trades stream; topbar shows positions + regime.
3. Stop mid-run → done pill, tear-sheet refreshes, Reports renders markdown,
   CSV/JSON export downloads a file.
4. Settings: change balance + risk + pairs → `PUT /api/config` persists
   (check `engine-server/user_config.yaml`) and applies next run; God Mode
   toggle stored.
5. Kill the engine → "Reconnecting…" banner appears; restart engine → banner
   clears, WS reconnects, data resumes.
6. Audit screen filters (reason/pair) work; live + audit rows merge.
7. Live check via CDP (see Phase 2 notes) that computed styles match tokens
   and no console errors.

### Pitfalls (carried forward from Phase 2)
- Two `<Router>`s crash → keep HashRouter only in `App.tsx`.
- CSS must be imported from TS (`main.tsx` imports `tokens.css`) — a raw
  `<link>` in index.html is dropped by Vite at build.
- Renderer console errors are visible via `--enable-logging` (filter binary
  junk: `tr -d '\000'`) or CDP `Runtime.evaluate`.
- Always restart the engine server after editing `engine-server/*.py`; health
  poll in main is 5s so UI catches up.
- `pkill -f` self-match: use bracket patterns or PID files.

### Phase 3 delivered + verified (Aug 1)

- All 10 tasks done. `npm run typecheck` + `npm run build` clean.
- **Live gap fixed in main/audit**: the engine WS was only persisting to SQLite —
  renderer never got live `signal/trade/equity/account` events. Added an
  `onEvent` forwarder in `AuditRecorder.handleMessage` (audit.ts) wired to
  `pushToRenderer` in main.ts. Verified live: RUNNING pill, live balance/equity/
  pnl/positions/regime in topbar, chart latest point == account equity, live
  signal ticker, sub-model agreement bars.
- **/api/equity shape**: returns flat `{equity, ts}` entries (not
  `{type,ts,data}`) — ipc.ts type + Dashboard mapping use `e.equity`.
- **Chart merge order**: store events are newest-first; merge iterates reversed
  so the curve stays ascending and the last point is the newest.
- **Settings persistence verified end-to-end**: changed balance to 60000 via the
  UI → persisted to `engine-server/user_config.yaml` + `GET /api/config`
  reports 60000. God Mode toggle uses `config.set({simulation:{god_mode}})` —
  engine reads it via `sim_cfg.get("god_mode", True)`.
- Equity metric falls back to last curve point when no live account event.
- App relaunch pattern: kill (`pkill -f "electron/dist/electro[n]"`) in a
  separate bash call, then launch in its own call, then wait ~15s for the
  DevTools page to appear — combining kill+launch in one call kept SIGTERM-ing
  the fresh process.
- Desktop logs: `/tmp/opencode/desktop12.log` (current), latest page ws in
  `/tmp/opencode/*.log` + curl `/json`.

---

## Phase 4 — Packaging, First-Run, and Production Hardening

**Goal:** an installable, professional desktop app for Linux/Windows/macOS
that connects to an externally-run engine. Pure Node.js inside the package;
the engine ships separately. Phase 3's screens, WS store, and audit chain are
the runtime foundation this phase wraps.

**Starting state:** Phase 3 complete and verified live. `desktop/` builds with
electron-vite 5 / vite 7 / react 19 / electron 43.2.0 / better-sqlite3 13.
Only deps are `better-sqlite3`, `react`, `react-dom`, `react-router-dom`, `ws`.
`main.ts` reads `ENGINE_URL` env → `~/.config/forex-desk/settings.json` →
default `http://127.0.0.1:8737`; audit DB at `~/.config/forex-desk/audit.db`.

**Decisions (simplicity first):**
- **electron-builder** is the only new production dep. `electron-updater` is
  wired behind a config flag but OFF by default (Track E).
- Tray, menu, window-state, log file, onboarding — all hand-rolled, no libs.
- This Linux box can produce + install-test **AppImage and deb**. Windows NSIS
  and mac DMG are properly configured + built in a GitHub Actions matrix
  (wine/macOS runners), since cross-building them here isn't reliable.
- **Update (2026-08-01):** the deb's xz compression gets OOM-killed on this
  7 GB box, so the deb (and its `dpkg -i` smoke test) is produced by the
  ubuntu-latest CI job instead. This box builds the AppImage only; dev-mode
  verification covers the identical main/renderer bundle.

### Ground rules carried forward
- **Isolation is law**: the desktop never spawns/imports Python. Track D's
  engine companion is detected and documented, never launched by the app.
- No new renderer deps unless a real need appears; UI stays dark/blue, calm.
- Rebuild/restart pitfalls from Phase 2/3 apply (see end of Phase 3 section):
  relaunch Electron via separate bash calls, `pkill -f "electron/dist/electro[n]"`.

### Track A — Installers

#### A1 electron-builder config
- `desktop/package.json` gains a `build` block and scripts
  `dist:linux` / `dist:win` / `dist:mac` / `dist` (current platform).
- appId `com.forexdesk.app`, productName `FOREX DESK`, executable name
  `forex-desk`.
- Targets:
  - Linux: AppImage + deb
  - Windows: NSIS (x64; oneClick false, allow to change install dir)
  - macOS: DMG (dmg.target dmg + zip fallback)
- `files`: include only `out/**` + `package.json` (main = `out/main/main.js`).
- `asar: true`. `npmRebuild: true` so better-sqlite3 matches each target ABI.
- `artifactName: FOREX-DESK-${version}-${os}-${arch}.${ext}`.
- Verify the built artifact contains NO `*.py`, `python`, `torch`, `engine-server`.

#### A2 Icons
- Generate a simple blue/white FOREX DESK mark: 512 master → 256/128/48/32 png,
  `build/icon.ico` (win), `build/icon.icns` (mac), `build/icon.png` (linux).
- Use ImageMagick `convert` if available; otherwise hand-write SVG → png via a
  tiny node script with no deps is NOT trivial — prefer `magick`/`convert`.

#### A3 Local build + install test (Linux)
- `npm run dist:linux` → artifacts in `desktop/dist/`.
- Test both: `./dist/*.AppImage --appimage-extract-and-run` (or chmod +x and
  run on `:0`) and `sudo dpkg -i dist/*.deb` then launch the installed binary.
- Confirm: window opens, connects to running engine, sim runs, audit persists,
  menu/tray from Track C work inside the packaged app.
- Report artifact sizes (target: single-digit MB, no Python inside).

#### A4 CI release matrix
- `.github/workflows/release.yml`: matrix [ubuntu-latest, windows-latest,
  macos-latest]; on tag `v*` (or manual dispatch) → `npm ci`, `npm run build`,
  `npm run dist:${os}`, upload `dist/**` as release assets (softprops/actions
  upload or gh release).
- On this box, only the Linux job can be run/verified; win/mac jobs are written
  to pass on their runners (electron-builder does the heavy lifting).
- Keep it simple: no code signing secrets in the first cut (note where signing
  hooks go later).

### Track B — First-run onboarding

#### B1 First-run detection
- In `main.ts`, before `createWindow`, read `settings.json`. If no
  `onboarded: true` flag, open the renderer at the onboarding route
  (hash `#/setup`) — simplest: a route in the existing HashRouter.
- Add `Setup` screen (`src/screens/Setup.tsx`): first route when not onboarded,
  reachable later from Settings ("Re-run onboarding" button).
- No extra IPC needed except what exists + one `app:set-onboarded` handler.

#### B2 Setup screen content (hand-rolled, matches tokens)
- Big title, calm copy: "Connect FOREX DESK to your engine service."
- ENGINE_URL input (defaults to current), **Connect** → calls `engine.health()`.
- Live status line: reachable / data ready / models loaded.
- If reachable but data not ready → **Prepare Data** button (reuses
  `data.prepare()`) + live `progress` message log (reuses `progress` events).
- If unreachable → inline card "Run the engine locally" with a copyable
  command `python engine-server/server.py` + a note about a bundled
  `forexdesk-engine` if detected nearby (Track D3).
- "Continue" enabled only when healthy+ready (or an explicit "Continue offline"
  link → existing banner takes over). On success: `app:set-onboarded` → navigate
  to `/`.

#### B3 Settings link
- Add a small "Run onboarding again" link at the bottom of the Settings Engine
  card that navigates to `#/setup` (resets nothing; re-runs the wizard).

### Track C — Production hardening (electron main)

#### C1 Tray
- `Tray` with the icon; tooltip = `FOREX DESK — {sim state}` (updated on
  `status`/`engine.status` changes).
- Context menu: Show / Run Simulation / Stop / Quit. Run/Stop reuse the same
  `sim.start`/`sim.stop` IPC handlers (add `tray:run`/`tray:stop` handlers or
  reuse existing ones from main directly — main can call the same fetch helper).
- Minimize-to-tray: intercept window `minimize` → `hide()` instead (setting
  toggle in Settings: "Minimize to tray", persisted in settings.json).
- Tray "Show" restores + focuses the window. Quit actually quits (sets a flag so
  `window-all-closed` doesn't keep it alive — on Linux the app quits anyway).

#### C2 App menu + About
- Build a native `Menu` (File: Run/Stop/Quit; View: reload/devtools; Help:
  About).
- About dialog: custom in-window modal route (`#/about`) or `dialog.showMessageBox`
  with version (from `app.getVersion()`), engine URL, health state, audit counts
  (from `auditStore.counts()`). Prefer the simple `showMessageBox`.

#### C3 Keyboard shortcuts
- Renderer-level (fastest): `useEffect` keydown in `Layout.tsx`:
  - `Ctrl/Cmd+R` → run sim (guard when engine offline)
  - `Ctrl/Cmd+.` → stop
  - `Ctrl/Cmd+1..5` → nav to the 5 screens (hash routes)
- Keep the native menu accelerators out of the way (don't double-bind Cmd+R on
  mac — document it).

#### C4 Window-state persistence
- Small module `windowState.ts` in main: save `{x,y,width,height,isMaximized}`
  to settings.json on close/move (debounced); restore on `createWindow`
  (validate bounds within a screen).

#### C5 Log file
- Add `log.ts` in main: append to `~/.config/forex-desk/main.log` (with
  timestamps). Wire: engine health poll transitions, WS connect/reconnect
  (audit recorder status), IPC handler errors, sim start/stop calls, uncaught
  `main`/`renderer` errors (`webContents.on("render-process-gone")`,
  `process.on("uncaughtException")`).
- Renderer console still captured via `--enable-logging` for dev.

#### C6 Audit DB hygiene
- In `AuditStore`: prune to the last N rows per table (default keep 20k trades,
  50k orders, 20k alerts) — run on open and on a timer while running.
- `pragma("wal_checkpoint(TRUNCATE)")` on `close()`.
- Keep the audit `.db`+`.db-wal`/`.db-shm` sizes bounded; report counts in About.

### Track D — Engine companion (deployed separately)

#### D1 PyInstaller build script
- `scripts/build_engine_binary.sh` (+ optional `engine-binary.spec`):
  `engine-server/server.py` entry; torch-free (engine stack imported lazily in
  the sim worker; venv already mocks torch/SB3). Output `dist/forexdesk-engine`.
- Build on the same OS family as the target (no cross-PyInstaller).

#### D2 Engine run docs
- `frontend/plan/DEPLOYMENT.md`: local run, `forexdesk-engine` run, systemd
  unit (Restart=always) and launchd plist examples, `ENGINE_URL` contract,
  health check, port note (default 8737).

#### D3 Desktop detection (never launch)
- On the Setup screen, probe a few conventional locations for a
  `forexdesk-engine` binary (next to the app, `$PATH`) — if found, show "Engine
  binary found — start it, then Connect." Detection only; no spawn, no shell
  exec. (Keep it minimal: a `$PATH` lookup via `which` is borderline; prefer a
  plain file check in the app dir + one env var `FOREXDESK_ENGINE_BIN`.)

### Track E — Updates (off by default)

#### E1 Version stamping
- electron-builder reads `package.json` version automatically; About shows it.
- Tag `vX.Y.Z` → CI release (Track A4).

#### E2 electron-updater (flag-gated)
- Add `electron-updater` dep; wrap in `updater.ts` that only activates when
  `settings.json` has `updates: { enabled: true, feedUrl }`. Default false.
- `checkForUpdates` on start + a Settings toggle ("Enable auto-update" + feed
  URL). No-op silently when disabled.
- First cut documents the contract; no live feed deployed.

### Verification (Phase 4 "done when")
1. `npm run dist:linux` → AppImage + deb build; **both install and launch on
   this box**, connect to the running engine, run a sim, audit persists after
   app restart.
2. Fresh profile (`rm -rf ~/.config/forex-desk`) → onboarding appears, connects,
   prepares data, proceeds; Settings "Re-run onboarding" works.
3. Tray (show/run/stop/quit), native menu + About, Ctrl+R/Ctrl+./Ctrl+1..5,
   window-state restore, `main.log` written, audit DB pruned and WAL-checkpointed.
4. Installer package contains zero Python/torch/engine files (grep the .AppImage
   after extraction).
5. `npm run typecheck` + `npm run build` clean; Linux CI job shape defined and
   locally equivalent to `dist:linux`.
6. Update flag is off; About shows version; toggling "Enable auto-update" is a
   no-op without a feed URL.

**Status (2026-08-01):**
1. AppImage builds here (141 MB) and its `linux-unpacked` is verified clean
   (0 `.py`, 0 `torch`, 0 `python` binaries; only `better-sqlite3` unpacked).
   The deb's xz compression OOMs on this 7 GB box → moved to the ubuntu-latest
   CI job (gzip is fine there); `dpkg -i` smoke test added to release.yml.
   Local AppImage *launch* on this box crashes in the GPU/network service
   (env issue); the same main/renderer bundle is verified every run in dev mode.
2. Done and verified in dev mode: fresh profile → 3-step onboarding wizard
   (engine URL + test, account balance, market pairs) → Get Started →
   dashboard; onboarding flag persisted; Settings "Re-run onboarding" works.
3. Done and verified: tray (Show/Run/Stop/Quit, tooltip sim-state),
   minimize-to-tray on close (with Settings toggle), native menu
   (Simulation Run/Stop/Quit, F12 devtools, Help → About with version+URL+
   audit counts), Ctrl+R run / Ctrl+. stop / Ctrl+1..5 nav (renderer-wins),
   window-state restore (saved bounds reapplied, `windowState` in settings.json),
   `main.log` written (health transitions, onboarding, tray, updater),
   audit prune + `wal_checkpoint(TRUNCATE)` on a 10-min timer + before-quit.
4. Verified (above).
5. `typecheck` + `build` green; release.yml matrix covers ubuntu/win/mac.
6. Updater off by default; Settings "Updates" toggle + feed URL persist to
   `settings.json`; `updater.apply()` is a silent no-op without a feed.

### Pitfalls / notes
- electron-builder needs network for cached download tools (NSIS/AppImage
  tooling) on first run.
- `better-sqlite3` is a native module: electron-builder `npmRebuild` must run;
  if packaged app fails on DB open, recheck rebuild + asar unpack
  (`asarUnpack: ["**/node_modules/better-sqlite3/**"]` if needed).
- Don't double-bind Ctrl/Cmd+R (menu accelerator vs renderer handler) — pick
  the renderer handler and drop the menu accelerator.
- Tray needs an icon file at runtime; ship it in `resources/` and load via
  `path.join(__dirname, ...)` — appPath differs in packaged vs dev.
- AppImage sandbox on this box: launch with `--no-sandbox` if the host lacks
  userns (matches dev setup).
- Keep `npm run typecheck` + `npm run build` green after every track.

---

## Phase 5 — Polish

- Onboarding, error states, exports, icons, LIVE MODE placeholder, perf pass.

---

## Log

| Date | Phase | What |
|---|---|---|
| 2026-07-31 | 0 | Scaffold + deps + health verified |
| 2026-08-01 | 1 | Config/state/engine/dataprep/reports/server written; data-prep end-to-end OK; live sim streaming all event types |
| 2026-08-01 | 1 | **Full 12,285-tick sim completed** (`done`); tear-sheet + CSV/JSON export verified on real data (end $34,832.94, -30.33%, max DD 31.94%, 1,035 trades, 44.5% win rate); Phase 1 closed |
| 2026-08-01 | 2 | Phase 2 detailed implementation plan written (toolchain, per-file spec, verification). Scaffold work begins here. |
| 2026-08-01 | 2 | **Phase 2 built + verified.** electron-vite 5 / vite 7 / react 19 / electron 43.2.0 / better-sqlite3 (Electron ABI OK). `npm run typecheck` + `npm run build` clean. Window opens on `:0`, connects to engine, audit recorder persisted live sim to `~/.config/forex-desk/audit.db` (trades/orders/alerts), survived app restart, desktop source has no `.py`. Phase 2 closed. |
| 2026-08-01 | 3 | **Phase 3 proposal approved; detailed plan written.** 10 tasks (equity IPC, EquityChart, topbar, dashboard, signals, trades, reports/markdown, settings, banners, design pass). |
| 2026-08-01 | 3 | **Phase 3 built + verified.** `typecheck`+`build` clean. Live gap fixed: `AuditRecorder` gained an `onEvent` forwarder → `pushToRenderer` (renderer now gets live signal/trade/equity/account events). `/api/equity` shape = flat `{equity,ts}` (fixed ipc type + mapping + chart merge order). Settings persistence proven end-to-end (balance 60000 via UI → `user_config.yaml`). Equity metric falls back to last curve point. Phase 3 closed. |
| 2026-08-01 | 4 | **Phase 4 proposal approved; detailed plan written.** Tracks A–E: installers (AppImage/deb here, NSIS/DMG via CI), icons, first-run onboarding wizard, tray/menu/shortcuts/window-state/log/audit hygiene, engine companion (PyInstaller, detection only), flag-gated updater. Implementation starts. |
| 2026-08-01 | 4 | **Phase 4 built.** A1 electron-builder config + A2 icons done. A3: AppImage built here (141 MB, `linux-unpacked` clean — 0 py/torch/python); deb's xz OOMs on this 7 GB box → moved to ubuntu-latest CI with a `dpkg -i` smoke test. B: 3-step onboarding wizard (engine/account/market), `app:get/set-onboarded` IPC, Settings "Re-run onboarding". C: tray + minimize-to-tray toggle, native menu + About (version/URL/audit counts), Ctrl+R/./1..5 shortcuts (renderer-wins, no double-bind), window-state persistence, `log.ts` wired (health transitions, sim, onboarding, updater), audit prune + WAL checkpoint timer. D: `scripts/build_engine_binary.sh` (PyInstaller), `DEPLOYMENT.md` (local + binary + systemd/launchd), `engine:detect-binary` IPC (env/app-dir probe, never launches) + Setup hint. E: version stamping via About; flag-gated `electron-updater` (`updater.ts` + Settings toggle/feed URL, silent no-op off). A4: `.github/workflows/release.yml` (ubuntu/win/mac matrix, `dpkg -i` smoke on Linux). `typecheck`+`build` green. Dev-mode verification complete: onboarding flow, shortcuts, minimize-to-tray, window restore, main.log. |
| 2026-08-01 | 5 | **Phase 5 "Black Ink, One Blue" design pass built.** Installed `@fontsource/{bodoni-moda,schibsted-grotesk,fragment-mono}`. Rewrote `tokens.css` (monochrome ink ramp, PnL-only colors, wordmark-only blue, font imports). Rewrote `app.css` (inverted white active nav chip, `.btn.primary` white plate, dark inputs, monochrome toggles/checkboxes, white-on-black chart, sticky tables, monochrome setup steps). Recolored `EquityChart.tsx` (white hero line, gray drawdown). Applied wordmark in `Layout.tsx` + `Setup.tsx`. CDP verified all 6 screens: Dashboard, Signals, Trades, Reports, Settings, Onboarding. `typecheck`+`build` green. |
