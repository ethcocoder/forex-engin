import os
import sys
from unittest.mock import MagicMock

# Define real classes for mock targets so type and issubclass checks don't crash
class MockTensor:
    pass

class MockModule:
    pass

class SmartTorchMock:
    Tensor = MockTensor
    device = MagicMock()
    __path__ = []
    def __getattr__(self, name):
        # Return a MagicMock for any other attribute, except special module attributes
        if name in ('__file__', '__loader__', '__name__', '__package__', '__spec__', '__path__'):
            return None
        return MagicMock()

sys.modules['torch'] = SmartTorchMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['stable_baselines3'] = MagicMock()

import time
import onnxruntime as ort
import pickle
import numpy as np
import pandas as pd
import logging
import structlog
from datetime import datetime

# Ensure root dir is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from configs.loader import load_config
from models.ensemble.aggregator import EnsembleAggregator
from models.onnx_inference import (
    ONNXTemporalWrapper,
    ONNXMAMLWrapper,
    ONNXRegimeWrapper,
    ONNXRLEnsembleWrapper,
    ONNXAttackerModel
)
from risk.risk_engine import RiskEngine, PortfolioState
from risk.sizing.kelly import KellySizer
from risk.limits.drawdown_limits import DrawdownFilter
from risk.limits.liquidity_filter import SpreadFilter
from execution.brokers.paper_broker import PaperBroker
from execution.execution_engine import ExecutionEngine
from infrastructure.trading_pipeline import TradingPipeline
from monitoring.performance_tracker import PerformanceTracker

# God Mode imports
from features.macro.deep_neural_synapse import DeepNeuralSynapse
from execution.routing.global_mesh_arbitrage import GlobalMeshArbitrage
from execution.hardware_offload.kernel_bypass_driver_integration import KernelBypassDriver

logger = structlog.get_logger()

class RealTimePipeline(TradingPipeline):
    """
    Extends TradingPipeline to perform automated exits based on dynamic holding decay steps
    and logs realized trades to the PerformanceTracker.
    """
    def __init__(self, ensemble, risk_engine, execution_engine, tracker, initial_capital=100000.0, 
                 synapse=None, mesh=None, attacker=None, kernel_bypass_driver=None):
        super().__init__(ensemble, risk_engine, execution_engine, initial_capital=initial_capital)
        self.tracker = tracker
        self.trade_durations = {}  # pair -> count of bars held
        self.max_hold_steps = 9    # Default signal decay steps
        
        # God Mode Components
        self.synapse = synapse
        self.mesh = mesh
        self.attacker = attacker
        self.kernel_bypass_driver = kernel_bypass_driver
        
    def process_tick(self, pair: str, X_window: np.ndarray, hour_ind: float, market_data: dict, current_bar_idx: int) -> None:
        actual_positions = self.execution_engine.sync_portfolio_state()
        self.portfolio_state.open_positions = actual_positions

        current_pos = actual_positions.get(pair, 0.0)
        
        # Track before state
        broker = self.execution_engine.broker
        cash_before = broker.cash
        pos_before = current_pos

        # Exit processing
        if current_pos != 0.0:
            self.trade_durations[pair] = self.trade_durations.get(pair, 0) + 1
            if self.trade_durations[pair] >= self.max_hold_steps:
                logger.info("Signal decay threshold reached. Initiating exit order.", pair=pair, steps=self.trade_durations[pair])
                direction = -1 if current_pos > 0 else 1
                from risk.risk_engine import OrderRequest
                exit_order = OrderRequest(pair=pair, direction=direction, size=abs(current_pos))
                
                # God Mode: Kernel Bypass for exit order
                if self.kernel_bypass_driver:
                    logger.info("God Mode: Sending exit order via Kernel Bypass.")
                    self.kernel_bypass_driver.send_raw_packet(b"\x01\x02\x03\x04_EXIT_ORDER")
                
                self.execution_engine.execute(exit_order, market_data)
                self.trade_durations[pair] = 0
                
                # Check exit pnl
                cash_after = broker.cash
                pos_after = broker.positions.get(pair, 0.0)
                if cash_after != cash_before:
                    realized_pnl = cash_after - cash_before
                    return_pct = realized_pnl / cash_before
                    self.update_pnl(realized_pnl, return_pct)
                    slippage = getattr(self.execution_engine, "last_execution_result", {}).get("slippage_pips", 0.0)
                    self.tracker.log_trade(
                        pair=pair,
                        direction=direction,
                        size=abs(pos_before - pos_after),
                        pnl=realized_pnl,
                        slippage_pips=slippage
                    )
                    self.tracker.update_equity(cash_after, time.time())
                return
        else:
            self.trade_durations[pair] = 0

        # Calculate unrealized pnl for RL environment wrapper state tracking
        unrealized = 0.0
        if current_pos != 0.0:
            entry = getattr(broker, "entry_prices", {}).get(pair, getattr(broker, "avg_entry", {}).get(pair, market_data["close"]))
            unrealized = current_pos * (market_data["close"] - entry)

        # God Mode: Deep Neural Synapse
        if self.synapse:
            synapse_data = pd.DataFrame({
                "USD_10Y": [market_data.get("usd_10y", 0)],
                "VIX": [market_data.get("vix", 0)],
                "COPPER": [market_data.get("copper", 0)],
                "GOLD": [market_data.get("gold", 0)],
                "S&P500": [market_data.get("sp500", 0)]
            })
            self.synapse.update_correlations(synapse_data)
            synapse_features = self.synapse.generate_synapse_features({"EURUSD": market_data["close"]})
            market_data.update(synapse_features)
            logger.debug("God Mode: Synapse features generated.", features=synapse_features)

        # God Mode: Global Mesh Arbitrage
        if self.mesh:
            simulated_market_state = {
                "NY4": {"EURUSD": market_data["close"]},
                "LD4": {"EURGBP": market_data["close"] * 0.85},
                "TY3": {"GBPUSD": market_data["close"] * 1.27}
            }
            opportunities = self.mesh.detect_triangular_opportunity(simulated_market_state)
            for opp in opportunities:
                logger.info("God Mode: Triangular arbitrage opportunity detected!", opportunity=opp)
                self.mesh.execute_mesh_trade(opp)

        # God Mode: Adversarial AI Scan
        if self.attacker:
            current_strategy = {"type": "Ensemble_Trading", "threshold": 0.00005}
            vulnerabilities = self.attacker.generate_adversarial_scenario(current_strategy)
            if vulnerabilities:
                logger.warning("God Mode: Adversarial AI detected vulnerabilities!", vulnerabilities=vulnerabilities)

        # 1. Dynamic Spread Gate
        spread_pips = market_data.get("spread_pips", 1.0)
        if current_pos == 0.0 and spread_pips > 1.5:
            return

        # 2. Extract Current HMM Regime from feature vector
        current_regime = int(np.argmax(X_window[0, -1, -4:]))

        # 3. Regime-Adaptive Signal Gating (Tuning entry threshold)
        regime_thresholds = {0: 0.0006, 1: 0.0018, 2: 0.0020, 3: 0.0012}
        self.ensemble.signal_generator.direction_threshold = regime_thresholds.get(current_regime, 0.0008)

        # Predict AlphaSignal using Master EnsembleAggregator
        signal = self.ensemble.predict(
            X_window,
            return_signal=True,
            current_position=current_pos,
            unrealized_pnl=unrealized,
            time_indicator=hour_ind,
            sample_idx=current_bar_idx,
            volatility=market_data.get("volatility", 0.0005),
            regime=current_regime
        )
        
        if signal.direction == 0:
            return

        # 4. Model Coherence Gate (Align Core Alpha and RL Policy)
        sub_preds = signal.metadata.get("sub_model_predictions", {})
        temporal_pred = sub_preds.get("temporal", 0.0)
        rl_pred = sub_preds.get("rl", 0.0)
        if current_pos == 0.0 and temporal_pred * rl_pred < 0:
            return

        # Prevent adding to an existing position in the same direction to avoid timer resets
        if current_pos != 0.0 and np.sign(current_pos) == signal.direction:
            return

        # Gate signal through Risk Engine
        order = self.risk_engine.gate(signal, pair, self.portfolio_state, market_data)
        
        # Execute Order
        if order is not None:
            success = False
            # God Mode: Kernel Bypass for entry order
            if self.kernel_bypass_driver:
                logger.info("God Mode: Sending entry order via Kernel Bypass.")
                self.kernel_bypass_driver.send_raw_packet(b"\x01\x02\x03\x04_ENTRY_ORDER")
                success = self.execution_engine.execute(order, market_data)
            else:
                success = self.execution_engine.execute(order, market_data)

            if success:
                logger.info("Real-Time order executed successfully", pair=pair, direction=order.direction, size=order.size)
                self.trade_durations[pair] = 0
                
                # 5. Regime-Adaptive Decay Steps (Hold longer in trends, exit quickly in ranges)
                regime_decay_steps = {0: 12, 1: 4, 2: 15, 3: 5}
                self.max_hold_steps = regime_decay_steps.get(signal.regime, signal.expected_decay_steps)
                
                # Check entry/reversal pnl
                cash_after = broker.cash
                pos_after = broker.positions.get(pair, 0.0)
                if cash_after != cash_before:
                    realized_pnl = cash_after - cash_before
                    return_pct = realized_pnl / cash_before
                    self.update_pnl(realized_pnl, return_pct)
                    slippage = getattr(self.execution_engine, "last_execution_result", {}).get("slippage_pips", 0.0)
                    self.tracker.log_trade(
                        pair=pair,
                        direction=order.direction,
                        size=abs(pos_before - pos_after),
                        pnl=realized_pnl,
                        slippage_pips=slippage
                    )
                    self.tracker.update_equity(cash_after, time.time())


class RegimeEnsembleWrapper:
    def __init__(self, regime_model=None, regime_mean=None, regime_std=None, hmm_features=None, features_df=None, seq_len=60):
        pass
    def predict(self, X, **kwargs):
        # Extract pre-calculated regime columns
        return X[:, -1, -4:]


class ONNXRegimeEnsembleEstimator:
    """Wrapper that combines GaussianHMM (hmmlearn) and ONNX LSTM classifier without PyTorch."""
    def __init__(self, hmm_path="saved_models/regime_ensemble.pkl.hmm", lstm_path="saved_models/regime_ensemble.lstm.onnx"):
        with open(hmm_path, "rb") as f:
            hmm_state = pickle.load(f)
        self.hmm = hmm_state["model"]
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.lstm_session = ort.InferenceSession(lstm_path, sess_options=opts)
        self.lstm_input_name = self.lstm_session.get_inputs()[0].name
        
        # Load weights
        with open("saved_models/regime_ensemble.pkl", "rb") as f:
            state = pickle.load(f)
        self.w_hmm = state.get("w_hmm", 0.5)
        self.w_lstm = state.get("w_lstm", 0.5)
        
    def predict(self, X, return_proba=True):
        X_hmm = X[:, -1, :]
        p_hmm = self.hmm.predict_proba(X_hmm)
        
        p_lstm_logits = self.lstm_session.run(None, {self.lstm_input_name: X.astype(np.float32)})[0]
        # Softmax
        exp_logits = np.exp(p_lstm_logits - np.max(p_lstm_logits, axis=-1, keepdims=True))
        p_lstm = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        p_combined = self.w_hmm * p_hmm + self.w_lstm * p_lstm
        
        eps = 1e-15
        p_combined = np.clip(p_combined, eps, 1.0 - eps)
        p_combined = p_combined / np.sum(p_combined, axis=-1, keepdims=True)
        
        if return_proba:
            return p_combined
        return np.argmax(p_combined, axis=-1)


def run_onnx_paper_trading(features_path="data/EUR_USD_features.csv", raw_path="data/EUR_USD_ticks.csv", god_mode: bool = True):
    # Configure logs
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger().setLevel(logging.INFO)
    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True
    )
    
    logger.info("Initializing run context (ONNX Engine mode)", god_mode=god_mode)
    logger.info("Starting High-Fidelity Real Paper Trading Simulator (ONNX Optimized)...")
    
    # 1. Load Data
    features_df = pd.read_csv(features_path, index_col="timestamp", parse_dates=True)
    raw_df = pd.read_csv(raw_path, index_col="timestamp", parse_dates=True)
    
    # Align indexes
    common_idx = raw_df.index.intersection(features_df.index)
    raw_df = raw_df.loc[common_idx]
    features_df = features_df.loc[common_idx]
    
    # 2. Load Config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
    app_config = load_config(config_path)
    config = app_config.model_dump() if hasattr(app_config, "model_dump") else app_config.dict()
    
    # 3. Load Regime Ensemble (ONNX style)
    logger.info("Loading ONNX Regime Ensemble...")
    regime_model = ONNXRegimeEnsembleEstimator()
    
    with open("saved_models/regime_feature_scaler.pkl", "rb") as f:
        regime_scaler = pickle.load(f)
        regime_mean = regime_scaler["mean"]
        regime_std = regime_scaler["std"]
        
    regime_cfg = config.get("models", {}).get("regime", {})
    hmm_features = regime_cfg.get("hmm_features", ["volatility_cc", "mean_reversion_hurst", "trend_strength", "vpin"])
    
    regime_features_df = features_df[hmm_features]
    regime_features_scaled = (regime_features_df.values - regime_mean) / regime_std
    regime_features_scaled = np.nan_to_num(regime_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    from numpy.lib.stride_tricks import sliding_window_view
    seq_len = 60
    windows = sliding_window_view(regime_features_scaled, window_shape=(seq_len, regime_features_scaled.shape[1]))
    X_regime = windows.squeeze(1)
    probs = regime_model.predict(X_regime, return_proba=True)
    
    # Pad probs to match features_df length
    padding = np.tile(probs[0], (seq_len - 1, 1))
    aligned_probs = np.vstack([padding, probs])
    
    regime_cols = [f"regime_{i}" for i in range(probs.shape[1])]
    for i, col in enumerate(regime_cols):
        features_df[col] = aligned_probs[:, i]
        
    exclude = ["timestamp", "close", "open", "high", "low", "volume", "bid", "ask"] + regime_cols
    features_cols = [col for col in features_df.columns if col not in exclude]
    
    # 4. Load Feature Scalers
    with open("saved_models/feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        scaler_mean = scaler["mean"]
        scaler_std = scaler["std"]
        
    raw_feature_indices = [features_df.columns.get_loc(col) for col in features_cols]
    
    # 5. Load all models and aggregator
    logger.info("Initializing Master Neural Ensemble Aggregator...")
    agg = EnsembleAggregator(config=config)
    agg.load("saved_models/ensemble_aggregator")
    
    # Initialize ONNX Wrappers
    temporal_wrapper = ONNXTemporalWrapper()
    temporal_wrapper.set_feature_indices(raw_feature_indices)
    
    maml_wrapper = ONNXMAMLWrapper()
    maml_wrapper.set_feature_indices(raw_feature_indices)
    
    regime_wrapper = RegimeEnsembleWrapper()
    
    rl_wrapper = ONNXRLEnsembleWrapper()
    rl_wrapper.set_config(features_cols, regime_cols)
    
    # Register models in aggregator (mocked torch handles is_torch flags)
    agg.register_model("temporal", temporal_wrapper, is_torch=True)
    agg.register_model("maml", maml_wrapper, is_torch=True)
    agg.register_model("regime", regime_wrapper, is_torch=False)
    agg.register_model("rl", rl_wrapper, is_torch=False)
    
    # Override from config
    ensemble_cfg = config.get("models", {}).get("ensemble", {})
    if "direction_threshold" in ensemble_cfg:
        agg.direction_threshold = ensemble_cfg["direction_threshold"]
        agg.signal_generator.direction_threshold = ensemble_cfg["direction_threshold"]
        
    # Setup Live Trading Pipeline components
    initial_capital = 10000.0
    pair = "EURUSD"
    
    # Sizer & Filters
    risk_cfg = config.get("risk", {})
    risk_engine = RiskEngine(config=risk_cfg)
    
    sizing_cfg = risk_cfg.get("sizing", {})
    kelly_frac = sizing_cfg.get("kelly_fraction", 0.15)
    max_risk = sizing_cfg.get("max_account_risk_pct", 0.02)
    risk_engine.set_sizer(KellySizer(fraction=kelly_frac, max_risk_pct=max_risk))
    
    cb_cfg = risk_cfg.get("circuit_breakers", {})
    if "daily_drawdown_limit" in cb_cfg:
        risk_engine.register_limit(DrawdownFilter(max_daily_dd=cb_cfg["daily_drawdown_limit"]))
        
    risk_engine.register_filter(SpreadFilter(default_max_spread_pips=4.0))
    
    # Broker & Execution (using C++ fast route DLL automatically)
    broker = PaperBroker(config={"initial_capital": initial_capital})
    execution_engine = ExecutionEngine(broker=broker)
    
    tracker = PerformanceTracker(initial_capital=initial_capital)
    
    # God Mode Component Initialization (using ONNX models)
    synapse = None
    mesh = None
    attacker = None
    kernel_bypass_driver = None
    
    if god_mode:
        try:
            synapse = DeepNeuralSynapse()
            mesh = GlobalMeshArbitrage()
            attacker = ONNXAttackerModel()
            kernel_bypass_driver = KernelBypassDriver("sfn0")
            kernel_bypass_driver.load_driver()
            logger.info("God Mode components initialized (ONNX Mode)")
        except Exception as e:
            logger.warning("God Mode initialization failed, disabling", error=str(e))
            
    # Initialize pipeline
    pipeline = RealTimePipeline(
        ensemble=agg,
        risk_engine=risk_engine,
        execution_engine=execution_engine,
        tracker=tracker,
        initial_capital=initial_capital,
        synapse=synapse,
        mesh=mesh,
        attacker=attacker,
        kernel_bypass_driver=kernel_bypass_driver
    )
    
    # Extract arrays for tick-by-tick feed
    closes = raw_df["close"].values
    timestamps = raw_df.index
    
    features_arr = features_df.copy().values
    features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    
    start_tick = seq_len - 1
    total_ticks = len(features_arr)
    
    logger.info("Starting tick-by-tick simulation loop...", total_ticks=total_ticks - start_tick)
    
    t_start = time.perf_counter()
    prev_date = None
    prev_week = None
    prev_month = None
    
    for i in range(start_tick, total_ticks):
        timestamp = timestamps[i]
        close = closes[i]
        
        # ── Periodic P&L Reset at Day/Week/Month Boundaries ──────────────
        cur_date = timestamp.date()
        cur_week = timestamp.isocalendar()[1]
        cur_month = timestamp.month
        
        if prev_date is not None and cur_date != prev_date:
            reset_weekly = (prev_week is not None and cur_week != prev_week)
            reset_monthly = (prev_month is not None and cur_month != prev_month)
            pipeline.reset_periodic_pnl(
                reset_daily=True,
                reset_weekly=reset_weekly,
                reset_monthly=reset_monthly
            )
        
        prev_date = cur_date
        prev_week = cur_week
        prev_month = cur_month
        
        # Slide window for the ensemble: shape [1, seq_len, total_features]
        X_input = features_arr[i - seq_len + 1 : i + 1]
        X_input = np.expand_dims(X_input, axis=0)
        
        hour_ind = timestamp.hour / 23.0
        
        # Rolling volatility
        if i >= 20:
            recent_closes = closes[i - 19 : i + 1]
            log_rets = np.diff(np.log(recent_closes))
            rolling_vol = float(np.std(log_rets))
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
            "usd_10y": 0.0, "vix": 0.0, "copper": 0.0, "gold": 0.0, "sp500": 0.0
        }
        
        broker.update_market_state({pair: market_data})
        pipeline.process_tick(pair, X_input, hour_ind, market_data, i - seq_len + 1)
        
        mtm_equity = broker.cash
        for pos_pair, pos_size in broker.positions.items():
            entry_px = broker.entry_prices.get(pos_pair, close)
            mtm_equity += pos_size * (close - entry_px)
            
        if (i - start_tick) % 1000 == 0:
            logger.info("Real-Time Paper Account State",
                        timestamp=timestamp.isoformat(),
                        cash=broker.cash,
                        mtm_equity=round(mtm_equity, 2),
                        open_positions=broker.get_positions())
            tracker.update_equity(mtm_equity, timestamp.timestamp())
            
    # Final mark-to-market
    final_equity = broker.cash
    for pos_pair, pos_size in list(broker.positions.items()):
        entry_px = broker.entry_prices.get(pos_pair, closes[-1])
        realized_pnl = pos_size * (closes[-1] - entry_px)
        final_equity += realized_pnl
        
        direction = 1 if pos_size > 0 else -1
        tracker.log_trade(
            pair=pos_pair,
            direction=direction,
            size=abs(pos_size),
            pnl=realized_pnl,
            slippage_pips=0.0
        )
        
    tracker.update_equity(final_equity, timestamps[-1].timestamp())
    t_elapsed = time.perf_counter() - t_start
    
    print("\n" + "="*80)
    print("           REAL-TIME HIGH-FIDELITY ONNX PAPER TRADING PERFORMANCE REPORT")
    print("="*80)
    print(tracker.generate_tear_sheet())
    print(f"Total Simulation Elapsed Time: {t_elapsed:.2f} seconds")
    print(f"Mean speed per tick: {t_elapsed / (total_ticks - start_tick) * 1e6:.1f} microseconds")
    print("="*80 + "\n")
    
    if kernel_bypass_driver:
        kernel_bypass_driver.unload_driver()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run High-Fidelity ONNX Paper Trading Simulator")
    parser.add_argument("--features", type=str, default="data/EUR_USD_features.csv", help="Path to features CSV")
    parser.add_argument("--raw", type=str, default="data/EUR_USD_ticks.csv", help="Path to raw ticks CSV")
    parser.add_argument("--disable-god-mode", action="store_true", help="Disable God Mode components")
    args = parser.parse_args()
    
    run_onnx_paper_trading(features_path=args.features, raw_path=args.raw, god_mode=not args.disable_god_mode)
