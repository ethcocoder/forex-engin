# FOREX DESK — desktop app (pure Node.js)

Two processes run side by side in dev:

1. **Engine service** — a standalone FastAPI Python service.
   Start it with:

   ```
   python engine-server/server.py
   ```

   It binds to `127.0.0.1:8737` by default (override with `ENGINE_HOST`
   and `ENGINE_PORT` env vars).

2. **Desktop app** — an Electron app (built in later phases) that connects
   to the engine service via its API only.

The engine service is a separate process and never ships inside the desktop
app; the desktop only talks to it over HTTP.
