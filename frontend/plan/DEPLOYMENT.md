# FOREX DESK — Deployment

FOREX DESK ships as **two independent pieces**:

| Piece | Tech | Runs where | How they connect |
|---|---|---|---|
| `desktop/` | Electron (pure Node.js) | User's machine | REST + WebSocket to the engine |
| `engine-server/` | Python / FastAPI | User's machine or a server | Listens on `127.0.0.1:8737` by default |

The desktop app **never** imports, spawns, or bundles the engine — it connects
over HTTP only. You can run the engine anywhere the desktop can reach it.

---

## 1. Run the engine locally (Python)

```bash
# one-time setup
bash engine-server/deps.sh            # creates .venv with all deps (torch-free)

# start the service
engine-server/.venv/bin/python engine-server/server.py
# → serving on http://127.0.0.1:8737
```

Health check:

```bash
curl http://127.0.0.1:8737/api/health
# {"status":"idle","version":"...","data_ready":false,"models_loaded":false}
```

First run: in the desktop app, Settings → Data → **Prepare Data** (downloads
`data/*.csv`), then **Run** in the top bar.

## 2. Standalone engine binary (no Python on the host)

PyInstaller bundles the engine into a single executable:

```bash
scripts/build_engine_binary.sh        # → dist/forexdesk-engine (same-OS only)
./dist/forexdesk-engine               # same server on :8737
```

> The binary only runs on the OS family it was built on. It does **not** bundle
> market data — that is downloaded at runtime via Prepare Data.

## 3. Desktop app

- **Linux:** `desktop/dist/FOREX-DESK-*.AppImage` (chmod +x, run) or
  `sudo dpkg -i FOREX-DESK-*.deb` (installs to `/opt/FOREX DESK/`).
- **Windows / macOS:** built by CI on tag `v*` (see `.github/workflows/release.yml`).
- First launch opens the onboarding wizard (engine URL, account, market).

## 4. The `ENGINE_URL` contract

The desktop reads the engine address in this order:

1. `ENGINE_URL` environment variable
2. `~/.config/forex-desk/settings.json` → `engineUrl` (set from Settings / onboarding)
3. Default `http://127.0.0.1:8737`

Required surface the desktop uses:

- `GET  /api/health` — polled every 5 s (connectivity + `data_ready`)
- `GET  /api/config`, `PUT /api/config`
- `POST /api/sim/start`, `POST /api/sim/stop`, `GET /api/sim/status`
- `GET  /api/equity`, `GET /api/trades`, `GET /api/signals`
- `GET  /api/reports/tear-sheet`, `GET /api/reports/export`
- `POST /api/data/prepare`
- `WS   /ws` — live events: `status`, `progress`, `account`, `positions`,
  `signal`, `trade`, `order`, `alert`, `equity`

## 5. Running the engine as a service

**systemd** (`/etc/systemd/system/forexdesk-engine.service`):

```ini
[Unit]
Description=FOREX DESK engine
After=network.target

[Service]
ExecStart=/home/user/forex-engin/dist/forexdesk-engine
WorkingDirectory=/home/user/forex-engin
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now forexdesk-engine
```

**launchd** (macOS, `~/Library/LaunchAgents/com.forexdesk.engine.plist`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
  <key>Label</key><string>com.forexdesk.engine</string>
  <key>ProgramArguments</key>
  <array><string>/Users/me/forex-engin/dist/forexdesk-engine</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.forexdesk.engine.plist
```

## 6. Notes

- Default port is **8737** (`ENGINE_HOST`/`ENGINE_PORT` env vars override).
- Point the desktop at a remote engine by changing the engine URL in the
  onboarding wizard or Settings → Engine → ENGINE URL.
- The onboarding screen shows "Engine binary found — start it, then Connect"
  when it detects a `forexdesk-engine` beside the app or in
  `$FOREXDESK_ENGINE_BIN` (detection only — the desktop never starts it).
