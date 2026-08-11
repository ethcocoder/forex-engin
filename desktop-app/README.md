# Forex Engin Desktop Command Center

The `desktop-app` directory contains the Electron + React + Tailwind desktop control plane for the `elite10x-pr` engine. It is intentionally **paper-mode first**: the renderer exposes monitoring and validation controls, while the secure Electron main process exposes only fixed IPC handlers for engine status and the local C++ chaos test binary.

## Run locally

```bash
cd desktop-app
pnpm install
pnpm dev
```

For a production renderer build:

```bash
pnpm build
pnpm start
```

The C++ validation bridge looks for:

```text
../cpp_engine/build/elite10x_engine
```

relative to the project root. Build that binary from the repository root before using the **Run chaos test** control if it is not already present.

## Product capabilities

The desktop shell includes a bilingual English/Amharic interface, persistent light/dark theme switching, paper-mode status, risk and system health panels, market-watch telemetry, audit events, and a validation output console. Live broker credentials are not bundled and no live order execution is exposed through the renderer by default.

## Safety notes

This is a control-plane prototype, not a guarantee of trading performance. Before connecting any broker, add authenticated paper-account adapters, order confirmation gates, stale-data watchdogs, fill reconciliation, audit logging, and an explicit live-mode enablement flow that requires deliberate operator confirmation.
