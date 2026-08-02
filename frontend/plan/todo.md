# FOREX DESK — End-to-End Task Checklist

Status: ⬜ Not started | 🔄 In progress | ✅ Done

**Architecture:** Two independent deliverables.
- `engine-server/` — standalone Python service (owns all maths/workflow).
- `desktop/` — Electron app in pure Node.js (displays state, records audit).
- They communicate **only via API** (REST + WebSocket). The desktop never
  imports, spawns, or bundles the engine.

---

## Phase 0 — Scaffold

- [x] 0.1 Create `engine-server/` stub files (`server.py`, `engine.py`, `state.py`, `reports.py`, `dataprep.py`, `requirements.txt`)
- [x] 0.2 Create `desktop/` tree (`frontend/electron/`, `frontend/src/`, `package.json`)
- [x] 0.3 Write `engine-server/requirements.txt` (fastapi, uvicorn, onnxruntime, numpy, pandas, pydantic, pyyaml — no torch)
- [x] 0.4 Implement minimal `engine-server/server.py`: FastAPI on `127.0.0.1:8737` (`ENGINE_HOST`/`ENGINE_PORT`)
- [x] 0.5 Add `GET /` root route (app name + status)
- [x] 0.6 Add `GET /api/health` → `{ status, version }`
- [x] 0.7 Create placeholder `desktop/package.json` (name, version)
- [x] 0.8 Add `.gitignore`: `engine-server/.venv/`, `frontend/node_modules/`, `dist/`, `build/`
- [x] 0.9 Add `desktop/README.md` note: run engine + desktop side by side
- [x] 0.10 Verify: start engine service, hit `/api/health`, get 200
- [x] 0.11 Verify: `desktop/` tree contains no Python files
- [x] 0.12 Verify: `git status` shows only intended new files

**Done when:** engine service boots and answers `/api/health`; desktop tree is pure Node.js.

---

## Phase 1 — Engine service

### 1.1 Configuration service
- [x] 1.1.1 Create `engine-server/config.py`
- [x] 1.1.2 Implement `get_config()` (load engine YAML via `configs.loader.load_config` from repo root)
- [x] 1.1.3 Implement `update_config(partial)` → merge + persist to `engine-server/user_config.yaml`
- [x] 1.1.4 Store `initial_capital` (balance) in config, default 10,000
- [x] 1.1.5 Engine service is source of truth for engine-affecting config

### 1.2 Simulation engine
- [x] 1.2.1 Create `engine-server/engine.py`
- [x] 1.2.2 Implement data readiness check (`data/EUR_USD_ticks.csv`, `data/EUR_USD_features.csv`) → `error: DATA_MISSING`
- [x] 1.2.3 Build engine objects (regime ensemble, scalers, model wrappers, aggregator, risk, paper broker, execution, tracker, pipeline) reusing `scripts/run_quantized_paper_trading.py` as a black box
- [x] 1.2.4 Run tick loop in a background thread
- [x] 1.2.5 Emit `account` + `equity` + `progress` after each `process_tick`
- [x] 1.2.6 Emit `signal` on each alpha signal
- [x] 1.2.7 Emit `order` on each execution (incl. fill, slippage, latency)
- [x] 1.2.8 Emit `trade` on each closed trade (incl. `exit_reason`)
- [x] 1.2.9 Emit `alert` on risk/system events (black swan, drawdown breach, convex sizing, regime, BMA weights)
- [x] 1.2.10 Implement graceful stop via `stop_event`, final `done` status + tear sheet

### 1.3 State store
- [x] 1.3.1 Create `engine-server/state.py` (thread-safe)
- [x] 1.3.2 Track `status` (`idle|running|done|error`) + `progress`
- [x] 1.3.3 Track `account` + `positions`
- [x] 1.3.4 Bounded ring buffers (5,000) for `signals`, `trades`, `orders`, `alerts`, `equity`
- [x] 1.3.5 Query helpers for REST history endpoints

### 1.4 WebSocket broadcaster
- [x] 1.4.1 Add `WS /ws` in `server.py`
- [x] 1.4.2 Enforce one event schema: `{ type, ts, data }` for all 8 types
- [x] 1.4.3 Send snapshot (`account`, `positions`, latest events) on connect
- [x] 1.4.4 Fan-out broadcast, no per-client state

### 1.5 Data prep
- [x] 1.5.1 Create `engine-server/dataprep.py`
- [x] 1.5.2 Implement `prepare()` → run `scripts/download_data.py`
- [x] 1.5.3 Run `scripts/generate_features.py --input data/EUR_USD_ticks.csv --output data/EUR_USD_features.csv`
- [x] 1.5.4 Stream subprocess output to WS as `progress`
- [x] 1.5.5 Flip `data_ready` in `/api/health` on success

### 1.6 REST surface
- [x] 1.6.1 `GET /api/health` (+ `data_ready`, `models_loaded`)
- [x] 1.6.2 `GET /api/config`
- [x] 1.6.3 `PUT /api/config`
- [x] 1.6.4 `POST /api/sim/start` (optional config overrides in body)
- [x] 1.6.5 `POST /api/sim/stop`
- [x] 1.6.6 `GET /api/sim/status` (status + progress + speed)
- [x] 1.6.7 `GET /api/trades?limit=`
- [x] 1.6.8 `GET /api/signals?limit=`
- [x] 1.6.9 `GET /api/equity`
- [x] 1.6.10 `GET /api/reports/tear-sheet`
- [x] 1.6.11 `GET /api/reports/export?format=csv|json`
- [x] 1.6.12 `POST /api/data/prepare`
- [x] 1.6.13 Verify full service with `curl` — no desktop involvement

**Done when:** sim streams all 8 event types; stop exits cleanly; config persists; data prep produces both CSVs; service is fully API-testable.

---

## Phase 2 — Electron shell

### 2.1 Main process
- [x] 2.1.1 Create `desktop/frontend/electron/main.ts`
- [x] 2.1.2 Read `ENGINE_URL` (default `http://127.0.0.1:8737`) from env/settings
- [x] 2.1.3 Poll engine `/api/health` until ready (timeout + failure UI)
- [x] 2.1.4 Create BrowserWindow (1440×900, `backgroundColor: '#0B0F14'`)
- [x] 2.1.5 Never spawn/kill the engine — connect only
- [x] 2.1.6 Clean quit: close window, WS closes on its own

### 2.2 Audit store
- [x] 2.2.1 Create `desktop/frontend/electron/audit.ts` (better-sqlite3)
- [x] 2.2.2 Create `trades` table
- [x] 2.2.3 Create `orders` table
- [x] 2.2.4 Create `alerts` table
- [x] 2.2.5 WS client in main subscribes to trade/order/alert and inserts rows
- [x] 2.2.6 IPC handlers: `audit:trades`, `audit:orders`, `audit:alerts`
- [x] 2.2.7 Verify history survives app restart

### 2.3 Preload
- [x] 2.3.1 Create `desktop/frontend/electron/preload.ts` with `contextBridge`
- [x] 2.3.2 Expose `config.get/set`
- [x] 2.3.3 Expose `sim.start/stop/status`
- [x] 2.3.4 Expose `reports.tearSheet/export`
- [x] 2.3.5 Expose `audit.trades/orders/alerts`
- [x] 2.3.6 Expose `data.prepare`
- [x] 2.3.7 Expose `engine.health/setUrl`
- [x] 2.3.8 Expose `on(event, cb)` WS subscription

### 2.4 Scaffold
- [x] 2.4.1 Init electron-vite + React + TypeScript template
- [x] 2.4.2 Add design tokens
- [x] 2.4.3 Add routing shell + empty screen stubs

**Done when:** window opens and connects to a running engine; trades persist in SQLite across restarts; desktop runs with no Python installed.
- [x] Verify: app launched on `:0`, "FOREX DESK" window up, `/api/health` green, audit.db grew (109→339 trades during sim), survived app restart, `find desktop -name "*.py" -not -path "*/node_modules/*"` empty

---

## Phase 3 — React UI

### 3.1 Design tokens
- [ ] 3.1.1 Create `desktop/frontend/src/styles/tokens.css` with exact palette (`Design.md` §4)
- [ ] 3.1.2 Inter + JetBrains Mono fonts
- [ ] 3.1.3 Radius 4px, 1px hairlines, spacing scale

### 3.2 Shell
- [ ] 3.2.1 Left sidebar (5 nav items, blue active state)
- [ ] 3.2.2 Top status bar: balance · equity · daily PnL · open positions · regime
- [ ] 3.2.3 Persistent `SIMULATION` badge
- [ ] 3.2.4 Run / stop control

### 3.3 Screens
- [ ] 3.3.1 Dashboard: equity curve chart (blue line + drawdown shading)
- [ ] 3.3.2 Dashboard: metric cards (Sharpe · Max DD · Win Rate · Trades)
- [ ] 3.3.3 Dashboard: live signal ticker
- [ ] 3.3.4 Signals: streaming table (time, direction, magnitude, confidence, regime, sub-model agreement)
- [ ] 3.3.5 Trades: trades table (pair, direction, size, entry/exit, colored pnl, slippage, exit reason, hold time)
- [ ] 3.3.6 Trades: filters (by reason, by pair) + order log below
- [ ] 3.3.7 Reports: rendered tear sheet (markdown)
- [ ] 3.3.8 Reports: export buttons (CSV, JSON download)
- [ ] 3.3.9 Settings: account balance input (persisted via `PUT /api/config`)
- [ ] 3.3.10 Settings: market pair selection
- [ ] 3.3.11 Settings: risk fields (kelly fraction, max risk %, circuit breakers)
- [ ] 3.3.12 Settings: engine section (`ENGINE_URL` + connection status, broker paper, God Mode toggle)
- [ ] 3.3.13 Settings: data readiness + "Prepare Data" with progress

### 3.4 State handling
- [ ] 3.4.1 Single WS client + simple store (no heavy state lib)
- [ ] 3.4.2 Monospace numerals everywhere; pnl colored
- [ ] 3.4.3 Loading / empty / error states on every data view
- [ ] 3.4.4 WS drop → calm "engine disconnected" banner + auto-reconnect

**Done when:** every screen updates live during a sim; settings persist via the engine; UI is calm, dark, blue+white only.

---

## Phase 4 — Packaging

> Status (2026-08-01): **DELIVERED.** Tracked in `work.md` (Tracks A–E). Key
> deviations: deb built on ubuntu-latest CI (local box OOMs on xz), AppImage
> built + contents-verified locally, dev-mode verification covers onboarding/
> tray/menu/shortcuts/window-state/log/updater.

### 4.1 Desktop installer
- [x] 4.1.1 Configure electron-builder in `desktop/package.json` (appId, productName `FOREX DESK`)
- [x] 4.1.2 Windows → NSIS installer
- [x] 4.1.3 macOS → DMG
- [x] 4.1.4 Linux → AppImage + deb
- [x] 4.1.5 Package contains only Node.js (no Python, no torch, no engine files)

### 4.2 Engine deployment (separate)
- [x] 4.2.1 Document local run: `python engine-server/server.py`
- [x] 4.2.2 Optional: PyInstaller bundle `forexdesk-engine` for no-Python machines
- [x] 4.2.3 Do NOT bundle data files (downloaded via Prepare Data)

### 4.3 Runtime wiring
- [x] 4.3.1 First run: prompt for `ENGINE_URL` (default localhost), verify `/api/health`
- [x] 4.3.2 Unreachable engine → clear connection issue in Settings
- [x] 4.3.3 Desktop never launches the engine

**Done when:** each platform installer builds and connects to a running engine; desktop install size stays small.

---

## Phase 5 — Black Ink, One Blue (IN PROGRESS)

> Monochrome design system: black/white chrome, green/red only on PnL figures,
> blue only in the wordmark (sidebar, onboarding lockup, icon/tray).
> Three-voice typography: Bodoni Moda (wordmark), Schibsted Grotesk (interface),
> Fragment Mono (numbers).

### 5A Typography + tokens
- [x] 5A.1 Install `@fontsource/{bodoni-moda,schibsted-grotesk,fragment-mono}`
- [x] 5A.2 Rewrite `tokens.css` — monochrome ink ramp (`--ink-0/1/2`, `--line`, `--paper`, `--gray`, `--faint`), PnL-only colors (`--pos`/`--neg`), wordmark-only `--brand`, font imports, `.wordmark`/`.eyebrow`/`.num`/`.muted` classes, focus-visible, reduced-motion

### 5B Monochrome shell
- [x] 5B.1 Rewrite `app.css` — inverted white active nav chip, white-outline buttons (`.btn.primary` white plate), dark inputs with hairline borders, monochrome toggles/checkboxes, white-on-black equity chart, sticky-header tables, monochrome setup steps, scrollbars

### 5C Components
- [x] 5C.1 Recolor `EquityChart.tsx` — `#2e7cf6` line/gradient → `var(--paper)` white hero line, drawdown shading → `var(--gray)`
- [x] 5C.2 Apply wordmark (`FOREX<span className="b">DESK</span>`) in `Layout.tsx` sidebar and `Setup.tsx` lockup
- [x] 5C.3 Fix `Setup.tsx` setup-step dots to monochrome (CSS handles; no blue)

### 5D Verification (CDP)
- [x] 5D.1 Dashboard — white chart line, gray labels, inverted nav chip, wordmark blue DESK
- [x] 5D.2 Signals — monochrome filters, gray empty state
- [x] 5D.3 Trades — green/red PnL only, mono table, hairline dividers
- [x] 5D.4 Reports — outline Export buttons, markdown headings sans-serif
- [x] 5D.5 Settings — monochrome toggles/checkboxes/inputs, App + Updates cards
- [x] 5D.6 Onboarding 3 steps — monochrome dots, blue wordmark only, white primary buttons

**Done when:** all screens monochrome; blue appears only in wordmark; PnL green/red is the only data color; fonts bundled via @fontsource.

---

## Cross-cutting rules (apply to every phase)

1. **Isolation is law.** The desktop never imports, spawns, or bundles the engine. API only.
2. Simplicity first — prefer a loop/function over a new library.
3. No torch in the engine service.
4. Engine math only in the engine service; renderer only formats.
5. One event schema `{ type, ts, data }` everywhere.
6. Commit per phase; each phase's done-block is the PR checklist.
