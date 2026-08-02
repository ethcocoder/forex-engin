# FOREX DESK — Desktop Application Design

## Status
- **Type:** Design plan
- **Branch:** `elite-pro`
- **Version:** 0.2 (draft)
- **Date:** 2026-07-31

---

## 1. Vision

A professional, cross-platform desktop application for Windows, Linux, and macOS
built with **Electron + React + TypeScript in pure Node.js**.

The trading **engine is a fully isolated, standalone Python service** that the
desktop never embeds. The two sides communicate **only through an API** — never
by import, never by bundling. This frees the desktop to be pure JavaScript, and
lets the engine run anywhere (localhost today, a remote/cloud gateway tomorrow).

The engine does **all** the maths and workflow. The frontend **displays** state
and **records** the audit and reports. No business logic lives in the UI.

> Today: paper simulation with existing funds.
> Future: the same app becomes the real gateway for balance checking and live
> trading. The design must therefore be calm, honest, and trustworthy.

**Core principle: simplicity.** Few screens, few dependencies, no fantasy UI.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 Desktop App (Electron · Node.js)              │
│  Main process ─ SQLite audit store · NO Python embedded       │
│  Renderer (React + TS) ── 5 screens, dark professional UI     │
└───────────────▲───────────────────────────────┬──────────────┘
        REST (controls/config)          WebSocket (live stream)
┌───────────────┴───────────────────────────────▼──────────────┐
│           Engine Service (standalone Python process)          │
│  Runs independently — owns models, maths, simulation loop     │
│  (EnsembleAggregator · RiskEngine · PaperBroker ·             │
│   ExecutionEngine · PerformanceTracker · RealTimePipeline)    │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 Why isolation (engine = separate service)

- **Language freedom:** the desktop is pure Node.js; the engine stays Python
  because that is where the quant stack lives.
- **API-only contract:** no imports, no bundling, no stdout parsing. The REST
  + WebSocket contract is the *only* seam between them.
- **Deployment freedom:** engine runs on localhost now, on a remote/cloud host
  later — the desktop connects to a configurable URL (`ENGINE_URL`).
- **Clean future gateway:** the same API contract is reused when the engine
  becomes the real-money gateway. The UI never changes its data source.
- Structured real-time state via WebSocket replaces fragile stdout parsing.

### 2.2 Engine-service deployment

- The engine is a standalone Python FastAPI process, started independently
  (`python engine-server/server.py`), bound to `127.0.0.1:8737` by default.
- The desktop does **not** spawn or kill it. It simply connects to
  `ENGINE_URL`. Local dev runs both processes side by side.
- Config that affects the engine (balance, risk, pairs) is sent **to** the
  engine via `PUT /api/config`; the engine is the source of truth.

### 2.3 Why torch-free engine

The quantized ONNX path already mocks `torch` / `stable_baselines3`.
The engine service therefore runs on `onnxruntime + numpy + pandas` only —
small, portable, and packaging-friendly.

### 2.4 Lifecycle

1. Engine service starts independently, loads config + models, exposes
   `/api/health` on its configured URL.
2. Desktop app launches; main process reads `ENGINE_URL` and checks health.
3. Renderer connects over WebSocket and subscribes to live events.
4. User starts/stops a simulation from the UI; engine streams events.
5. Electron main records every trade/order/alert into SQLite.
6. Each side shuts down on its own; the desktop never tears down the engine.

---

## 3. Backend data contract

### 3.1 REST endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | backend alive, models loaded, data ready |
| GET | `/api/config` | full current config |
| PUT | `/api/config` | update settings (balance, risk, pairs) |
| POST | `/api/sim/start` | start simulation |
| POST | `/api/sim/stop` | stop simulation |
| GET | `/api/sim/status` | `idle \| running \| done \| error`, progress, µs/tick |
| GET | `/api/trades` | closed trades history (`?limit=`) |
| GET | `/api/signals` | signal history (`?limit=`) |
| GET | `/api/equity` | equity curve points |
| GET | `/api/reports/tear-sheet` | metrics + markdown summary |
| GET | `/api/reports/export` | audit/report export (`?format=csv\|json`) |
| POST | `/api/data/prepare` | run `download_data.py` → `generate_features.py` |

### 3.2 WebSocket events

One schema for every message: `{ "type": string, "ts": number, "data": object }`.

#### `account` — account & portfolio state
Source: `PortfolioState` (`risk/risk_engine.py:37`) + `PaperBroker`.
```json
{
  "type": "account",
  "ts": 1780000000,
  "data": {
    "initial_capital": 10000.0,
    "cash": 10123.45,
    "equity": 10130.12,
    "daily_pnl": 130.12,
    "weekly_pnl": 200.0,
    "monthly_pnl": -50.0,
    "win_rate": 0.55,
    "win_loss_ratio": 1.4
  }
}
```

#### `signal` — live alpha signal
Source: `AlphaSignal` (`models/ensemble/signal_generator.py:12`).
```json
{
  "type": "signal",
  "ts": 1780000000,
  "data": {
    "direction": 1,
    "magnitude": 0.4,
    "confidence": 0.72,
    "uncertainty": 0.3,
    "expected_decay_steps": 7,
    "regime": 2,
    "sub_models": { "temporal": 0.0012, "maml": 0.0008, "rl": 0.9 },
    "threshold": 0.0018
  }
}
```

#### `order` — order + fill
Source: `OrderRequest` (`risk/risk_engine.py:50`) + `last_execution_result`.
```json
{
  "type": "order",
  "ts": 1780000000,
  "data": {
    "id": "o_01",
    "pair": "EURUSD",
    "direction": 1,
    "size": 0.15,
    "order_type": "MARKET",
    "stop_loss": 1.0842,
    "status": "FILLED",
    "fill_price": 1.0851,
    "slippage_pips": 0.8,
    "latency_ns": 2500,
    "meta": { "anti_fragile": true, "vol_z_score": 0.3, "multiplier": 1.0 }
  }
}
```

#### `trade` — closed trade (audit trail)
```json
{
  "type": "trade",
  "ts": 1780000000,
  "data": {
    "pair": "EURUSD",
    "direction": -1,
    "size": 0.15,
    "entry_price": 1.0851,
    "exit_price": 1.0872,
    "pnl": 31.5,
    "slippage_pips": 1.2,
    "hold_steps": 9,
    "exit_reason": "stop_loss"
  }
}
```
`exit_reason ∈ { stop_loss, decay, drawdown, reversal }`.

#### `alert` — risk/system/model events
```json
{
  "type": "alert",
  "ts": 1780000000,
  "level": "critical",
  "source": "risk",
  "code": "BLACK_SWAN",
  "message": "Extreme volatility regime shift.",
  "data": { "z_score": 4.7 }
}
```
Known codes: `BLACK_SWAN` (risk_engine.py:135), `DRAWDOWN_BREACH`
(pipeline), `CONVEX_SIZING` (risk_engine.py:161), `REGIME_SHIFT`,
`BMA_WEIGHTS` (pipeline `_reinforce_bma`).

#### `equity` — equity point for the live chart
```json
{ "type": "equity", "ts": 1780000000, "data": { "equity": 10130.12 } }
```

#### `progress` — simulation progress
```json
{
  "type": "progress",
  "ts": 1780000000,
  "data": { "tick": 12345, "total": 50000, "speed_us": 212.5 }
}
```

### 3.3 Tear-sheet metrics (Reports screen)

Generated by `PerformanceTracker.generate_tear_sheet()`
(`monitoring/performance_tracker.py:41`):

- Initial Capital, Ending Capital, Total Return %
- Max Drawdown %
- Raw Sharpe Ratio
- Total Trades, Win Rate %, Win/Loss Ratio
- Total Slippage Drag (pips)
- Equity curve (for chart)

### 3.4 Config surfaced in Settings

From `configs/config.yaml` + `configs/loader.py` (Pydantic `AppConfig`).

| Section | Editable fields | Default |
|---|---|---|
| Account | initial balance | 10,000 |
| Market | pairs | EUR_USD |
| Risk | sizing method, kelly fraction, max account risk %, circuit breakers (daily/weekly/monthly DD) | kelly / 0.15 / 0.0075 / 0.03·0.06·0.10 |
| Engine | `ENGINE_URL`, broker (`paper`), God Mode on/off | `http://127.0.0.1:8737` / paper / on |
| Data | status + "Prepare Data" action | — |

---

## 4. Frontend design system

### 4.1 Principles

- **Calm & professional.** No gradients, glows, or rounded cards.
- **Trust.** Persistent `SIMULATION` badge in the top bar. Honest labels,
  explicit numbers, no decoration.
- **Simplicity.** 5 screens, one accent color, hairline dividers.

### 4.2 Palette — three roles only

| Role | Color | Hex |
|---|---|---|
| Base background | near-black | `#0B0F14` |
| Raised surface | | `#121820` |
| Hairline border | | `#1E2833` |
| Text primary | white | `#F5F7FA` |
| Text muted | | `#8A94A6` |
| Brand accent | blue (single accent) | `#2E7CF6` |
| Positive | green (PnL only) | `#22C55E` |
| Negative | red (PnL only) | `#EF4444` |
| Warning | amber (alerts only) | `#F59E0B` |

Semantic colors (green/red/amber) are used **exclusively** for their meaning.

### 4.3 Typography

- UI: **Inter**
- Numerals/figures: **JetBrains Mono** (tabular, trustworthy numbers)
- Corner radius: 4px. Dividers: 1px hairlines. Generous whitespace.

### 4.4 Layout

- Left sidebar navigation (5 items, brand-blue active state)
- Top status bar: balance · equity · daily PnL · open positions · regime ·
  `SIMULATION` badge · run/stop control
- Content area on a raised surface grid

---

## 5. Screens

### 5.1 Dashboard
- Top status bar (account summary)
- Equity curve chart — brand-blue line, drawdown shading
- Metric cards: Sharpe · Max DD · Win Rate · Trades
- Live signal ticker (last signal, direction, confidence, regime)

### 5.2 Signals
- Streaming table: time, direction, magnitude, confidence, regime,
  sub-model agreement

### 5.3 Trades (Audit)
- Trades table: entry/exit, pnl, slippage, exit reason, hold time + filters
- Order log below

### 5.4 Reports
- Rendered tear sheet (markdown)
- Export buttons: CSV, JSON

### 5.5 Settings
- Account (balance) / Market / Risk / Engine / Data sections
- Engine section: `ENGINE_URL` (default `http://127.0.0.1:8737`) + connection status
- Data readiness status + "Prepare Data" (download + feature generation)

---

## 6. Project structure

```
desktop/                     # Electron app — pure Node.js, no Python
├── frontend/
│   ├── electron/            # main + preload (SQLite audit)
│   └── src/                 # React + TS (electron-vite), 5 screens, tokens
└── package.json             # electron-builder config

engine-server/               # Standalone Python API service (separate process)
├── server.py                # FastAPI app, WS broadcaster
├── engine.py                # wraps run_quantized_paper_trading internals
├── state.py                 # live state store (account, signals, trades, alerts)
├── reports.py               # tear-sheet + CSV/JSON export
├── dataprep.py              # download + feature-generation runner
└── requirements.txt         # fastapi, uvicorn, onnxruntime, + engine deps
```

The two trees are **independent**. `desktop/` never imports Python; `engine-server/`
never ships inside the desktop. Existing engine (`models/`, `risk/`, `execution/`,
`infrastructure/`) is **untouched**; the engine server imports and wraps it.

---

## 7. Implementation phases

| Phase | Scope | Done when |
|---|---|---|
| 0 | Scaffold `desktop/` + `engine-server/`, gitignore, README | Engine server boots, `/api/health` OK |
| 1 | Engine server: config load/save, sim start/stop, WS events, data-prep | Full sim streams live events over API |
| 2 | Electron shell: connect to engine via API, lifecycle, SQLite audit | App opens and talks to engine over HTTP/WS |
| 3 | React UI: design system + 5 screens | All screens live off WS/REST |
| 4 | Packaging: desktop = electron-builder only; engine deployed separately | Installer builds; connects to external engine |
| 5 | Polish: exports, error states, empty-data onboarding, real-gateway placeholder | v1 ready |

---

## 8. Key decisions & risks

| # | Item | Decision / Note |
|---|---|---|
| 1 | Engine isolation | desktop = pure Node.js; engine = separate Python service; API-only seam |
| 2 | Engine URL | `ENGINE_URL` configurable; default `http://127.0.0.1:8737` |
| 3 | Data files missing | `data/` is empty; "Prepare Data" step is mandatory for v1 |
| 4 | `onnxruntime` missing | not in `requirements.txt`; must be added to engine-server deps |
| 5 | Compiled `.dll` speedups | Windows-only; code falls back to Python — keep fallback |
| 6 | Torch-free engine path | quantized ONNX path mocks torch → small engine, small desktop |
| 7 | Audit persistence | Electron main writes trades/orders/alerts to SQLite |
| 8 | Future real-money gateway | same REST/WS contract; engine deploys remotely, desktop unchanged |

---

## 9. Out of scope (v1)

- Live/market data streaming, real brokers
- Training / retraining UI (backend scripts only for now)
- Multi-user / cloud sync
- Mobile build
