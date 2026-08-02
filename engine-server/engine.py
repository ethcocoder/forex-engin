"""In-process simulation engine for FOREX DESK.

Wraps the quantized ONNX paper-trading loop from
``scripts/run_quantized_paper_trading.py`` without touching any engine source.

The engine-service venv is torch-free: ``torch`` / ``stable_baselines3`` are
mocked before any engine module is imported (the sim path runs on quantized
ONNX + numpy/pandas). The heavy engine modules are imported lazily, inside the
worker thread, so the FastAPI service boots fast.

The engine runs in a background thread. Every observable event is pushed into
the ``StateStore`` and enqueued for WebSocket broadcast. Signals/orders/trades
are captured by wrapping the pipeline's collaborators (aggregator.predict,
execution_engine.execute, tracker.log_trade); alerts and exit reasons come from
the pipeline's structlog output, which is rendered as JSON so nothing is lost.
"""

import json
import logging
import os
import pickle
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Torch / stable-baselines3 mocks — installed before any engine import.
# ---------------------------------------------------------------------------
class MockTensor:
    pass


class MockModule:
    pass


class SmartTorchMock:
    Tensor = MockTensor
    device = MagicMock()
    __path__: Any = []

    def __getattr__(self, name: str) -> Any:
        if name in (
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__spec__",
            "__path__",
        ):
            return None
        return MagicMock()


sys.modules.setdefault("torch", SmartTorchMock())
sys.modules.setdefault("torch.nn", MagicMock())
sys.modules.setdefault("torch.nn.functional", MagicMock())
sys.modules.setdefault("stable_baselines3", MagicMock())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import structlog  # noqa: E402

from state import StateStore  # noqa: E402

ENGINE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ENGINE_ROOT / "data"
TICKS_FILE = DATA_DIR / "EUR_USD_ticks.csv"
FEATURES_FILE = DATA_DIR / "EUR_USD_features.csv"
MODELS_DIR = ENGINE_ROOT / "saved_models"
SEQ_LEN = 60
SNAPSHOT_EVERY = 5  # broadcast account/equity/positions/progress every N ticks

_components: Optional[Dict[str, Any]] = None
_components_lock = threading.Lock()

# Attribute keys (from structlog records) that are interesting as alert payload.
_LOG_ATTR_SKIP = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_text", "exc_info", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "logger", "event",
        "timestamp", "level",
    }
)


def _load_components() -> Dict[str, Any]:
    """Import the engine stack once (torch is already mocked)."""
    global _components
    with _components_lock:
        if _components is not None:
            return _components

        sys.path.insert(0, str(ENGINE_ROOT))
        sys.path.insert(0, str(ENGINE_ROOT / "scripts"))

        from configs.loader import load_config  # noqa: F401
        from models.ensemble.aggregator import EnsembleAggregator
        from models.onnx_inference import (  # noqa: F401
            ONNXTemporalWrapper,
            ONNXMAMLWrapper,
            ONNXRLEnsembleWrapper,
        )
        from risk.risk_engine import RiskEngine, OrderRequest
        from risk.sizing.kelly import KellySizer
        from risk.limits.drawdown_limits import DrawdownFilter
        from risk.limits.liquidity_filter import SpreadFilter
        from execution.brokers.paper_broker import PaperBroker
        from execution.execution_engine import ExecutionEngine
        from monitoring.performance_tracker import PerformanceTracker
        import run_quantized_paper_trading as engine_script

        components: Dict[str, Any] = {
            "EnsembleAggregator": EnsembleAggregator,
            "ONNXTemporalWrapper": ONNXTemporalWrapper,
            "ONNXMAMLWrapper": ONNXMAMLWrapper,
            "ONNXRLEnsembleWrapper": ONNXRLEnsembleWrapper,
            "RiskEngine": RiskEngine,
            "OrderRequest": OrderRequest,
            "KellySizer": KellySizer,
            "DrawdownFilter": DrawdownFilter,
            "SpreadFilter": SpreadFilter,
            "PaperBroker": PaperBroker,
            "ExecutionEngine": ExecutionEngine,
            "PerformanceTracker": PerformanceTracker,
            "script": engine_script,
        }

        # God Mode components — degrade gracefully if they fail to import.
        god_mode_sources = (
            ("DeepNeuralSynapse", "features.macro.deep_neural_synapse"),
            ("GlobalMeshArbitrage", "execution.routing.global_mesh_arbitrage"),
            ("KernelBypassDriver", "execution.hardware_offload.kernel_bypass_driver_integration"),
            ("ONNXAttackerModel", "models.onnx_inference"),
        )
        for name, module_name in god_mode_sources:
            try:
                components[name] = getattr(__import__(module_name, fromlist=[name]), name)
            except Exception:
                components[name] = None

        _components = components
        return _components


def _safe(value: Any) -> Any:
    """Make arbitrary attrs JSON-safe (np types, mocks, arrays -> str)."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        try:
            return str(value)
        except Exception:
            return "<unserializable>"


def _json_safe_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _safe(v) for k, v in data.items()}


class _EngineLogHandler(logging.Handler):
    """Parse the JSON-rendered structlog output for alerts + exit reasons.

    The sim pipeline logs warning/critical events via structlog (stdlib
    factory). We configure a JSON renderer, so every record's message is a JSON
    object we can parse without touching any engine code.
    """

    def __init__(self, outer: "SimulationEngine") -> None:
        super().__init__(level=logging.WARNING)
        self._outer = outer

    def emit(self, record: logging.LogRecord) -> None:
        outer = self._outer
        message = record.getMessage()
        attrs: Dict[str, Any] = {}
        event = message
        try:
            parsed = json.loads(message)
            if isinstance(parsed, dict):
                event = str(parsed.get("event", ""))
                attrs = {k: _safe(v) for k, v in parsed.items() if k not in _LOG_ATTR_SKIP}
        except (ValueError, TypeError):
            pass

        lowered = event.lower()
        if "stop-loss triggered" in lowered:
            outer._pending_exit_reason = "stop_loss"
        elif "drawdown limit breached" in lowered:
            outer._pending_exit_reason = "drawdown"
        elif "signal decay threshold" in lowered:
            outer._pending_exit_reason = "decay"

        level = getattr(record, "levelname", "warning").lower()
        code = "ENGINE"
        if "black swan" in lowered:
            code = "BLACK_SWAN"
        elif "drawdown" in lowered:
            code = "DRAWDOWN_BREACH"
        elif "convex sizing" in lowered:
            code = "CONVEX_SIZING"
        elif "regime" in lowered:
            code = "REGIME_SHIFT"
        elif "bma weights" in lowered:
            code = "BMA_WEIGHTS"

        outer.emit_alert(level, "engine", code, event, **attrs)


class SimulationEngine:
    """Owns one simulation run in a background thread."""

    def __init__(self, state: Optional[StateStore] = None, sink: Optional[Callable[[dict], None]] = None) -> None:
        self.state = state or StateStore()
        self._sink = sink  # callable(event: dict) — server wires the WS queue
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._config: Dict[str, Any] = {}
        self._initial_capital: float = 10000.0
        self._broker: Any = None
        self._pending_exit_reason: Optional[str] = None
        self._last_close_ctx: Dict[str, Any] = {}
        self._log_handler: Optional[_EngineLogHandler] = None

    # -- public control -------------------------------------------------------
    def start(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate data, then launch the sim in a background thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": False, "error": "ALREADY_RUNNING", "status": self.state.status}

            if not TICKS_FILE.exists() or not FEATURES_FILE.exists():
                self.state.reset()
                self.state.set_error("DATA_MISSING: run POST /api/data/prepare first")
                return {
                    "ok": False,
                    "error": "DATA_MISSING",
                    "status": "error",
                    "message": "Missing data/EUR_USD_ticks.csv or data/EUR_USD_features.csv",
                }

            self._config = dict(config or {})
            self._initial_capital = float(
                self._config.get("account", {}).get("balance", 10000.0)
            )
            self._stop_event = threading.Event()
            self._pending_exit_reason = None
            self._last_close_ctx = {}

            self.state.reset()
            self.state.set_status("loading")
            self._thread = threading.Thread(target=self._run, name="sim-engine", daemon=True)
            self._thread.start()
            self.emit_status("loading")
            return {"ok": True, "status": "loading"}

    def stop(self) -> Dict[str, Any]:
        """Request a clean stop and wait for the thread to finish."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self.state.set_status("idle")
                self.emit_status("idle")
                return {"ok": True, "status": "idle"}

            self._stop_event.set()
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                return {"ok": False, "status": "running", "error": "STOP_TIMEOUT"}

            if self.state.status != "done":
                self.state.set_status("done")
                self.emit_status("done")
            return {"ok": True, "status": self.state.status}

    def status(self) -> Dict[str, Any]:
        alive = bool(self._thread is not None and self._thread.is_alive())
        status = self.state.status
        if alive and status in ("idle", "done", "error"):
            status = "running"
        return {
            "status": status,
            "error": self.state.error,
            "progress": dict(self.state.progress),
            "running": alive,
        }

    # -- threading ------------------------------------------------------------
    def _run(self) -> None:
        try:
            # The engine stack resolves relative paths (saved_models/, data/)
            # against the repo root, exactly like scripts/run_quantized_paper_trading.py.
            try:
                os.chdir(ENGINE_ROOT)
            except OSError:
                pass
            self._setup_logging()
            self._build_and_run()
        except SystemExit:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            traceback.print_exc()
            self.state.set_error(f"{type(exc).__name__}: {exc}")
            self.emit_status("error", error=str(exc))

    def _setup_logging(self) -> None:
        """Replicate the script's log config but render structlog as JSON so the
        log handler can parse alerts + exit reasons out of it."""
        logging.basicConfig(level=logging.INFO, force=True)
        logging.getLogger().setLevel(logging.INFO)
        structlog.configure(
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
            processors=[
                structlog.stdlib.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.JSONRenderer(serializer=json.dumps),
            ],
        )
        handler = _EngineLogHandler(self)
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

    def _build_and_run(self) -> None:
        config = self._config
        components = _load_components()
        initial_capital = self._initial_capital

        try:
            regime_model, regime_scaler, aggregator = self._load_models(components, config)
        except FileNotFoundError as exc:
            self.state.set_error(f"MODEL_MISSING: {exc}")
            self.emit_status("error", error=str(exc))
            return

        self.state.set_models_loaded(True)

        raw_df, features_df = self._load_data()
        features_df, features_cols, regime_cols = self._compute_regime(
            components, config, features_df, regime_model, regime_scaler
        )
        raw_feature_indices = [features_df.columns.get_loc(col) for col in features_cols]

        pipeline = self._build_pipeline(
            components, config, initial_capital, aggregator,
            raw_feature_indices, features_cols, regime_cols,
        )
        self._broker = pipeline.execution_engine.broker
        self._run_tick_loop(pipeline, raw_df, features_df)

    # -- model / data assembly (mirrors run_quantized_paper_trading) ---------
    def _load_models(self, components: Dict[str, Any], config: Dict[str, Any]):
        script = components["script"]
        regime_model = script.ONNXRegimeEnsembleEstimator()
        with open(str(MODELS_DIR / "regime_feature_scaler.pkl"), "rb") as f:
            regime_scaler = pickle.load(f)
        aggregator = components["EnsembleAggregator"](config=config)
        aggregator.load("saved_models/ensemble_aggregator")
        return regime_model, regime_scaler, aggregator

    def _load_data(self):
        features_df = pd.read_csv(FEATURES_FILE, index_col="timestamp", parse_dates=True)
        raw_df = pd.read_csv(TICKS_FILE, index_col="timestamp", parse_dates=True)
        common_idx = raw_df.index.intersection(features_df.index)
        raw_df = raw_df.loc[common_idx]
        features_df = features_df.loc[common_idx]
        if len(features_df) < SEQ_LEN:
            raise RuntimeError(
                f"Not enough aligned bars ({len(features_df)}) for seq_len {SEQ_LEN}"
            )
        return raw_df, features_df

    def _compute_regime(self, components, config, features_df, regime_model, regime_scaler):
        from numpy.lib.stride_tricks import sliding_window_view

        regime_cfg = config.get("models", {}).get("regime", {})
        hmm_features = regime_cfg.get(
            "hmm_features", ["volatility_cc", "mean_reversion_hurst", "trend_strength", "vpin"]
        )
        regime_mean = regime_scaler["mean"]
        regime_std = regime_scaler["std"]

        regime_features_df = features_df[hmm_features]
        regime_scaled = (regime_features_df.values - regime_mean) / regime_std
        regime_scaled = np.nan_to_num(regime_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        if len(regime_scaled) < SEQ_LEN:
            raise RuntimeError(f"Not enough bars ({len(regime_scaled)}) for regime window")
        windows = sliding_window_view(regime_scaled, window_shape=(SEQ_LEN, regime_scaled.shape[1]))
        X_regime = windows.squeeze(1)
        probs = regime_model.predict(X_regime, return_proba=True)

        padding = np.tile(probs[0], (SEQ_LEN - 1, 1))
        aligned_probs = np.vstack([padding, probs])
        regime_cols = [f"regime_{i}" for i in range(probs.shape[1])]
        for i, col in enumerate(regime_cols):
            features_df[col] = aligned_probs[:, i]

        exclude = ["timestamp", "close", "open", "high", "low", "volume", "bid", "ask"] + regime_cols
        features_cols = [col for col in features_df.columns if col not in exclude]
        return features_df, features_cols, regime_cols

    def _build_pipeline(self, components, config, initial_capital, aggregator,
                        raw_feature_indices, features_cols, regime_cols):
        script = components["script"]

        temporal = components["ONNXTemporalWrapper"](model_path="saved_models/temporal_model.onnx")
        temporal.set_feature_indices(raw_feature_indices)
        maml = components["ONNXMAMLWrapper"](model_path="saved_models/quantized/maml_model_int8.onnx")
        maml.set_feature_indices(raw_feature_indices)
        regime_wrapper = script.RegimeEnsembleWrapper()
        rl = components["ONNXRLEnsembleWrapper"](model_path="saved_models/quantized/rl_agent_ppo_int8.onnx")
        rl.set_config(features_cols, regime_cols)

        aggregator.register_model("temporal", temporal, is_torch=True)
        aggregator.register_model("maml", maml, is_torch=True)
        aggregator.register_model("regime", regime_wrapper, is_torch=False)
        aggregator.register_model("rl", rl, is_torch=False)

        ensemble_cfg = config.get("models", {}).get("ensemble", {})
        if "direction_threshold" in ensemble_cfg:
            aggregator.direction_threshold = ensemble_cfg["direction_threshold"]
            aggregator.signal_generator.direction_threshold = ensemble_cfg["direction_threshold"]

        risk_cfg = config.get("risk", {})
        risk_engine = components["RiskEngine"](config=risk_cfg)
        sizing_cfg = risk_cfg.get("sizing", {})
        risk_engine.set_sizer(components["KellySizer"](
            fraction=sizing_cfg.get("kelly_fraction", 0.15),
            max_risk_pct=sizing_cfg.get("max_account_risk_pct", 0.02),
        ))
        cb_cfg = risk_cfg.get("circuit_breakers", {})
        if "daily_drawdown_limit" in cb_cfg:
            risk_engine.register_limit(components["DrawdownFilter"](
                max_daily_dd=cb_cfg.get("daily_drawdown_limit", 0.02),
                max_weekly_dd=cb_cfg.get("weekly_drawdown_limit", 0.04),
                max_monthly_dd=cb_cfg.get("monthly_drawdown_limit", 0.08),
            ))
        risk_engine.register_filter(components["SpreadFilter"](default_max_spread_pips=4.0))

        broker = components["PaperBroker"](config={"initial_capital": initial_capital})
        execution_engine = components["ExecutionEngine"](broker=broker)
        tracker = components["PerformanceTracker"](initial_capital=initial_capital)

        synapse = mesh = attacker = kernel_bypass_driver = None
        sim_cfg = config.get("simulation", {})
        if sim_cfg.get("god_mode", True):
            try:
                if components.get("DeepNeuralSynapse"):
                    synapse = components["DeepNeuralSynapse"]()
                if components.get("GlobalMeshArbitrage"):
                    mesh = components["GlobalMeshArbitrage"]()
                if components.get("ONNXAttackerModel"):
                    attacker = components["ONNXAttackerModel"](
                        model_path="saved_models/quantized/adversarial_attacker_int8.onnx"
                    )
                if components.get("KernelBypassDriver"):
                    kernel_bypass_driver = components["KernelBypassDriver"]("sfn0")
                    kernel_bypass_driver.load_driver()
            except Exception as exc:
                self.emit_alert("warning", "engine", "GOD_MODE_DISABLED", f"God Mode disabled: {exc}")

        return self._make_pipeline(
            script.RealTimePipeline, initial_capital, aggregator, risk_engine,
            execution_engine, tracker, synapse, mesh, attacker, kernel_bypass_driver,
        )

    def _make_pipeline(self, real_time_pipeline_cls, initial_capital, aggregator,
                       risk_engine, execution_engine, tracker, synapse, mesh,
                       attacker, kernel_bypass_driver):
        outer = self

        class BridgePipeline(real_time_pipeline_cls):
            def __init__(self):
                super().__init__(
                    ensemble=aggregator,
                    risk_engine=risk_engine,
                    execution_engine=execution_engine,
                    tracker=tracker,
                    initial_capital=initial_capital,
                    synapse=synapse,
                    mesh=mesh,
                    attacker=attacker,
                    kernel_bypass_driver=kernel_bypass_driver,
                )
                self._install_hooks()

            def _install_hooks(self):
                # -- signals: wrap aggregator.predict ----------------------
                ensemble = self.ensemble
                orig_predict = ensemble.predict

                def hooked_predict(X, **kwargs):
                    out = orig_predict(X, **kwargs)
                    outer.on_signal(out)
                    return out

                ensemble.predict = hooked_predict

                # -- orders: wrap execution_engine.execute -----------------
                ee = self.execution_engine
                orig_execute = ee.execute

                def hooked_execute(order, market_data=None):
                    broker = ee.broker
                    pos_before = broker.positions.get(order.pair, 0.0)
                    entry_before = broker.entry_prices.get(order.pair)
                    hold_steps = self.trade_durations.get(order.pair, 0)
                    started_ns = time.perf_counter_ns()
                    ok = orig_execute(order, market_data)
                    latency_us = (time.perf_counter_ns() - started_ns) / 1000.0
                    result = getattr(ee, "last_execution_result", None) or {}
                    if pos_before != 0.0:
                        outer._last_close_ctx = {
                            "entry_price": entry_before,
                            "fill_price": result.get("fill_price"),
                            "hold_steps": hold_steps,
                            "orig_dir": 1 if pos_before > 0 else -1,
                        }
                    outer.on_order(order, result, ok, latency_us, pos_before)
                    return ok

                ee.execute = hooked_execute

                # -- trades: wrap tracker.log_trade -------------------------
                tr = self.tracker
                orig_log_trade = tr.log_trade

                def hooked_log_trade(pair, direction, size, pnl, slippage_pips):
                    orig_log_trade(pair, direction, size, pnl, slippage_pips)
                    outer.on_trade(pair, size, pnl, slippage_pips)

                tr.log_trade = hooked_log_trade

        return BridgePipeline()

    # -- tick loop ------------------------------------------------------------
    def _run_tick_loop(self, pipeline, raw_df, features_df) -> None:
        closes = raw_df["close"].values
        timestamps = raw_df.index
        features_arr = features_df.copy().values
        features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        pair = self._config.get("simulation", {}).get("pair", "EURUSD")
        broker = pipeline.execution_engine.broker
        start_tick = SEQ_LEN - 1
        total_ticks = len(features_arr)
        total_run = total_ticks - start_tick

        self.state.set_progress({"tick": 0, "total": total_run, "speed_us": 0.0})
        self.state.set_status("running")
        self.emit_status("running")

        t_start = time.perf_counter()
        prev_date = prev_week = prev_month = None
        last_emit = 0
        i = start_tick

        while i < total_ticks:
            if self._stop_event.is_set():
                break

            timestamp = timestamps[i]
            close = float(closes[i])

            cur_date = timestamp.date()
            cur_week = timestamp.isocalendar()[1]
            cur_month = timestamp.month
            if prev_date is not None and cur_date != prev_date:
                pipeline.reset_periodic_pnl(
                    reset_daily=True,
                    reset_weekly=(prev_week is not None and cur_week != prev_week),
                    reset_monthly=(prev_month is not None and cur_month != prev_month),
                )
            prev_date, prev_week, prev_month = cur_date, cur_week, cur_month

            X_input = features_arr[i - SEQ_LEN + 1: i + 1]
            X_input = np.expand_dims(X_input, axis=0)

            hour_ind = timestamp.hour / 23.0

            if i >= 20:
                recent = closes[i - 19: i + 1]
                rolling_vol = float(np.std(np.diff(np.log(recent))))
            else:
                rolling_vol = 0.0005

            market_data = {
                "close": close,
                "price": close,
                "mid_price": close,
                "spread_pips": 0.75 + np.random.rand() * 0.5,
                "adv": 1000000.0,
                "pip_value": 0.0001,
                "volatility": rolling_vol,
                "usd_10y": 0.0,
                "vix": 0.0,
                "copper": 0.0,
                "gold": 0.0,
                "sp500": 0.0,
            }

            broker.update_market_state({pair: market_data})
            pipeline.process_tick(pair, X_input, hour_ind, market_data, i - SEQ_LEN + 1)

            tick_no = i - start_tick + 1
            if tick_no - last_emit >= SNAPSHOT_EVERY:
                last_emit = tick_no
                speed_us = (time.perf_counter() - t_start) / tick_no * 1e6
                self.emit_snapshots(pipeline, close, timestamp, tick_no, total_run, speed_us)

            i += 1

        self._finalize(pipeline, closes, timestamps)

    def _finalize(self, pipeline, closes, timestamps) -> None:
        broker = pipeline.execution_engine.broker
        tracker = pipeline.tracker

        # Mark remaining positions to market.
        self._pending_exit_reason = "final_mtm"
        final_equity = broker.cash
        for pos_pair, pos_size in list(broker.positions.items()):
            entry = broker.entry_prices.get(pos_pair, closes[-1])
            realized = pos_size * (closes[-1] - entry)
            final_equity += realized
            tracker.log_trade(
                pair=pos_pair,
                direction=1 if pos_size > 0 else -1,
                size=abs(pos_size),
                pnl=realized,
                slippage_pips=0.0,
            )
        tracker.update_equity(final_equity, timestamps[-1].timestamp())
        self._pending_exit_reason = None

        self.state.set_account(self._build_account(broker, pipeline, final_equity))
        self.state.set_progress({
            "tick": self.state.progress.get("tick", 0),
            "total": self.state.progress.get("total", 0),
            "speed_us": self.state.progress.get("speed_us", 0.0),
            "stopped_early": bool(self._stop_event.is_set()),
        })

        markdown = tracker.generate_tear_sheet()
        metrics = self._compute_metrics(tracker)
        self.state.set_reports({"markdown": markdown, "metrics": metrics})
        self.emit("report", {"markdown": markdown, "metrics": metrics})

        if getattr(pipeline, "kernel_bypass_driver", None) is not None:
            try:
                pipeline.kernel_bypass_driver.unload_driver()
            except Exception:
                pass

        self.state.set_status("done")
        self.emit_status("done")

    # -- snapshots ------------------------------------------------------------
    def _build_account(self, broker, pipeline, equity: float) -> Dict[str, Any]:
        return {
            "initial_capital": self._initial_capital,
            "cash": round(float(broker.cash), 2),
            "equity": round(float(equity), 2),
            "daily_pnl": round(float(pipeline.portfolio_state.daily_pnl), 2),
            "weekly_pnl": round(float(pipeline.portfolio_state.weekly_pnl), 2),
            "monthly_pnl": round(float(pipeline.portfolio_state.monthly_pnl), 2),
            "win_rate": float(pipeline.portfolio_state.win_rate),
            "win_loss_ratio": float(pipeline.portfolio_state.win_loss_ratio),
        }

    def emit_snapshots(self, pipeline, close: float, timestamp, tick: int, total: int, speed_us: float) -> None:
        broker = pipeline.execution_engine.broker

        unrealized = 0.0
        for pos_pair, pos_size in broker.positions.items():
            entry = broker.entry_prices.get(pos_pair, close)
            unrealized += pos_size * (close - entry)
        equity = broker.cash + unrealized

        account = self._build_account(broker, pipeline, equity)
        account["unrealized_pnl"] = round(float(unrealized), 2)
        self.state.set_account(account)

        positions = {}
        for pos_pair, pos_size in broker.positions.items():
            entry = broker.entry_prices.get(pos_pair, close)
            positions[pos_pair] = {
                "size": float(pos_size),
                "entry_price": float(entry),
                "mark_price": float(close),
                "unrealized_pnl": round(float(pos_size * (close - entry)), 2),
            }
        self.state.set_positions(positions)

        self.state.set_progress({"tick": tick, "total": total, "speed_us": round(float(speed_us), 1)})

        self.emit("account", account)
        self.emit("positions", positions)
        self.emit("equity", {"equity": round(float(equity), 2)})
        self.emit("progress", {"tick": tick, "total": total, "speed_us": round(float(speed_us), 1)})

    def _compute_metrics(self, tracker) -> Dict[str, Any]:
        equity = np.asarray(tracker.equity_curve, dtype=float)
        trades = list(tracker.trades)

        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            total_return_pct = float((equity[-1] / tracker.initial_capital - 1.0) * 100.0)
            running_max = np.maximum.accumulate(equity)
            drawdown = (running_max - equity) / running_max
            max_dd_pct = float(np.max(drawdown) * 100.0)
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns))
            sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
        else:
            total_return_pct = 0.0
            max_dd_pct = 0.0
            sharpe = 0.0

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate_pct = (len(wins) / len(trades) * 100.0) if trades else 0.0
        avg_win = float(np.mean([t["pnl"] for t in wins])) if wins else 0.0
        avg_loss = abs(float(np.mean([t["pnl"] for t in losses]))) if losses else 0.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        return {
            "initial_capital": tracker.initial_capital,
            "ending_capital": float(equity[-1]) if len(equity) else tracker.initial_capital,
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe": round(sharpe, 4),
            "total_trades": len(trades),
            "win_rate_pct": round(win_rate_pct, 1),
            "win_loss_ratio": round(win_loss_ratio, 2),
            "total_slippage_pips": round(float(tracker.total_slippage_pips), 2),
        }

    # -- event hooks ----------------------------------------------------------
    def on_signal(self, signal) -> None:
        if getattr(signal, "direction", 0) == 0:
            return
        meta = getattr(signal, "metadata", {}) or {}
        sub = meta.get("sub_model_predictions", {}) or {}
        ts = time.time()
        data = {
            "direction": signal.direction,
            "magnitude": round(float(signal.magnitude), 4),
            "confidence": round(float(signal.confidence), 4),
            "uncertainty": round(float(signal.uncertainty), 4),
            "expected_decay_steps": int(signal.expected_decay_steps),
            "regime": int(signal.regime),
            "sub_models": {k: round(float(v), 6) for k, v in sub.items()},
            "threshold": meta.get("dynamic_threshold"),
        }
        self.state.push_signal(data, ts)
        self._enqueue({"type": "signal", "ts": ts, "data": data})

    def on_order(self, order, result, ok: bool, latency_us: float, pos_before: float) -> None:
        result = result or {}
        status = result.get("status") or ("REJECTED" if not ok else "PENDING")
        ts = time.time()
        data = {
            "id": f"o_{int(ts * 1000)}",
            "pair": order.pair,
            "direction": order.direction,
            "size": order.size,
            "order_type": getattr(order, "order_type", "MARKET"),
            "stop_loss": order.stop_loss,
            "status": status,
            "fill_price": result.get("fill_price"),
            "slippage_pips": round(float(result.get("slippage_pips", 0.0)), 2),
            "latency_us": round(latency_us, 1),
            "meta": dict(getattr(order, "metadata", {}) or {}),
            "role": "entry" if pos_before == 0.0 else "exit",
        }
        self.state.push_order(data, ts)
        self._enqueue({"type": "order", "ts": ts, "data": data})

    def on_trade(self, pair: str, size: float, pnl: float, slippage_pips: float) -> None:
        ctx = self._last_close_ctx or {}
        reason = self._pending_exit_reason or ("reversal" if ctx else "final_mtm")
        ts = time.time()
        data = {
            "pair": pair,
            "direction": int(ctx.get("orig_dir", 1)),
            "size": size,
            "entry_price": ctx.get("entry_price"),
            "exit_price": ctx.get("fill_price"),
            "pnl": round(float(pnl), 2),
            "slippage_pips": round(float(slippage_pips), 2),
            "hold_steps": int(ctx.get("hold_steps", 0)),
            "exit_reason": reason,
        }
        self._pending_exit_reason = None
        self._last_close_ctx = {}
        self.state.push_trade(data, ts)
        self._enqueue({"type": "trade", "ts": ts, "data": data})

    def emit_alert(self, level: str, source: str, code: str, message: str, **attrs: Any) -> None:
        ts = time.time()
        data = {
            "level": level,
            "source": source,
            "code": code,
            "message": message,
            "data": _json_safe_dict(attrs),
        }
        self.state.push_alert(data, ts)
        self._enqueue({"type": "alert", "ts": ts, **data})

    def emit_status(self, status: str, **extra: Any) -> None:
        self.emit("status", {"status": status, **extra})

    def emit(self, type_: str, data: Dict[str, Any]) -> None:
        ts = time.time()
        ring = {
            "signal": "push_signal",
            "trade": "push_trade",
            "order": "push_order",
            "equity": "push_equity",
        }
        if type_ in ring:
            getattr(self.state, ring[type_])(data, ts)
        self._enqueue({"type": type_, "ts": ts, "data": data})

    def _enqueue(self, event: Dict[str, Any]) -> None:
        if self._sink is None:
            return
        try:
            self._sink(event)
        except Exception:
            pass
