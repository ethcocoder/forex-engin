
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys

# 1. Parse threads count before imports to configure thread environments
num_threads = None
for i, arg in enumerate(sys.argv):
    if arg.startswith("--threads="):
        num_threads = arg.split("=")[1]
    elif arg == "--threads" and i + 1 < len(sys.argv):
        num_threads = sys.argv[i + 1]

if num_threads is not None:
    try:
        threads_to_use = int(num_threads)
    except ValueError:
        threads_to_use = max(1, os.cpu_count() - 1) if os.cpu_count() else 4
else:
    # Use max cores - 1 by default to boost speed while keeping the OS responsive
    threads_to_use = max(1, os.cpu_count() - 1) if os.cpu_count() else 4

os.environ["OMP_NUM_THREADS"] = str(threads_to_use)
os.environ["MKL_NUM_THREADS"] = str(threads_to_use)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads_to_use)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads_to_use)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads_to_use)

import time
import pickle
import numpy as np
import pandas as pd
import torch
import gc

torch.set_num_threads(threads_to_use)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

import logging
import structlog
from datetime import datetime

# Ensure root dir is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from configs.loader import load_config
from models.temporal.combined import TemporalFusionModel
from models.meta_learner.maml import MAMLModel
from models.regime.combined import RegimeEnsembleEstimator
from models.rl_agent.ppo_agent import PPOModel
from models.ensemble.aggregator import EnsembleAggregator
from risk.risk_engine import RiskEngine, PortfolioState
from risk.sizing.kelly import KellySizer
from risk.sizing.fixed_fractional import FixedFractionalSizer
from risk.limits.drawdown_limits import DrawdownFilter
from risk.limits.liquidity_filter import SpreadFilter
from execution.brokers.paper_broker import PaperBroker
from execution.execution_engine import ExecutionEngine
from infrastructure.trading_pipeline import TradingPipeline
from monitoring.performance_tracker import PerformanceTracker

from scripts.run_backtest import MAMLEnsembleWrapper, TemporalEnsembleWrapper, RegimeEnsembleWrapper, RLEnsembleWrapper

# God Mode Imports
from features.macro.deep_neural_synapse import DeepNeuralSynapse
from execution.routing.global_mesh_arbitrage import GlobalMeshArbitrage
from execution.hardware_offload.kernel_bypass_driver_integration import KernelBypassDriver
from models.adversarial_ai.attacker_model import AttackerModel

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
        self.trade_durations = {} # pair -> count of bars held
        self.max_hold_steps = 9   # Default signal decay steps
        
        # God Mode Components
        self.synapse = synapse
        self.mesh = mesh
        self.attacker = attacker
        self.kernel_bypass_driver = kernel_bypass_driver
        
    def process_tick(self, pair: str, X_window: np.ndarray, hour_ind: float, market_data: dict, current_bar_idx: int) -> None:
        # 1. Sync Portfolio State with broker
        actual_positions = self.execution_engine.sync_portfolio_state()
        self.portfolio_state.open_positions = actual_positions

        current_pos = actual_positions.get(pair, 0.0)
        
        # Track before state
        broker = self.execution_engine.broker
        cash_before = broker.cash
        pos_before = current_pos

        # 2. Dynamic holding & exit processing
        if current_pos != 0.0:
            self.trade_durations[pair] = self.trade_durations.get(pair, 0) + 1
            if self.trade_durations[pair] >= self.max_hold_steps:
                # Signal holding period complete -> execute market exit
                logger.info("Signal decay threshold reached. Initiating exit order.", pair=pair, steps=self.trade_durations[pair])
                direction = -1 if current_pos > 0 else 1
                from risk.risk_engine import OrderRequest
                exit_order = OrderRequest(pair=pair, direction=direction, size=abs(current_pos))
                
                # God Mode: Kernel Bypass for exit order
                if self.kernel_bypass_driver:
                    logger.info("God Mode: Sending exit order via Kernel Bypass.")
                    # Simulate raw packet transmission
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

        # God Mode: Deep Neural Synapse for enhanced features
        if self.synapse:
            # Simulate high-res data for synapse update (using current market data as a proxy)
            # In a real scenario, this would be a stream of actual high-res data
            synapse_data = pd.DataFrame({
                "USD_10Y": [market_data.get("usd_10y", 0)],
                "VIX": [market_data.get("vix", 0)],
                "COPPER": [market_data.get("copper", 0)],
                "GOLD": [market_data.get("gold", 0)],
                "S&P500": [market_data.get("sp500", 0)]
            })
            self.synapse.update_correlations(synapse_data)
            synapse_features = self.synapse.generate_synapse_features({"EURUSD": market_data["close"]})
            # Augment market_data with synapse features
            market_data.update(synapse_features)
            logger.debug("God Mode: Synapse features generated.", features=synapse_features)

        # God Mode: Global Mesh Arbitrage
        if self.mesh:
            # Simulate market state for arbitrage detection
            # This would come from real-time feeds from different centers
            simulated_market_state = {
                "NY4": {"EURUSD": market_data["close"]},
                "LD4": {"EURGBP": market_data["close"] * 0.85}, # Placeholder
                "TY3": {"GBPUSD": market_data["close"] * 1.27}  # Placeholder
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

        # 3. Predict AlphaSignal using our real master EnsembleAggregator!
        signal = self.ensemble.predict(
            X_window,
            return_signal=True,
            current_position=current_pos,
            unrealized_pnl=unrealized,
            time_indicator=hour_ind,
            sample_idx=current_bar_idx,
            volatility=market_data.get("volatility", 0.0005)
        )
        
        if signal.direction == 0:
            return  # No entry action

        # Prevent adding to an existing position in the same direction to avoid timer resets
        if current_pos != 0.0 and np.sign(current_pos) == signal.direction:
            return

        # 4. Gate signal through Risk Engine
        order = self.risk_engine.gate(signal, pair, self.portfolio_state, market_data)
        
        # 5. Execute Order
        if order is not None:
            success = False
            # God Mode: Kernel Bypass for entry order
            if self.kernel_bypass_driver:
                logger.info("God Mode: Sending entry order via Kernel Bypass.")
                # Simulate raw packet transmission
                self.kernel_bypass_driver.send_raw_packet(b"\x01\x02\x03\x04_ENTRY_ORDER")
                success = self.execution_engine.execute(order, market_data) # Still need to execute through broker for state management
            else:
                success = self.execution_engine.execute(order, market_data)

            if success:
                logger.info("Real-Time order executed successfully", pair=pair, direction=order.direction, size=order.size)
                self.trade_durations[pair] = 0
                self.max_hold_steps = signal.expected_decay_steps
                
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


def run_real_paper_trading(features_path="data/EUR_USD_features.csv", raw_path="data/EUR_USD_ticks.csv", fast: bool = False, god_mode: bool = True):
    if fast:
        logging.basicConfig(level=logging.INFO, force=True)
        logging.getLogger().setLevel(logging.INFO)
        structlog.configure(
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True
        )
        logger.info("Initializing run context (fast mode)", threads_configured=threads_to_use, fast=fast)
        logger.info("Starting High-Fidelity Real Paper Trading Simulator (fast mode)...")
    else:
        logger.info("Initializing run context", threads_configured=threads_to_use)
        logger.info("Starting High-Fidelity Real Paper Trading Simulator with God Mode...")
    
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
    
    # 3. Load Regime Ensemble
    logger.info("Loading Regime Ensemble...")
    regime_model = RegimeEnsembleEstimator()
    regime_model.load("saved_models/regime_ensemble.pkl")
    
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
    
    features_arr = features_df.copy().values
    features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Create sliding windows only when not running in fast low-memory mode
    if not fast:
        windows = sliding_window_view(features_arr, window_shape=(seq_len, features_arr.shape[1]))
        X_master = np.copy(windows.squeeze(1))
    else:
        X_master = None
    
    # 5. Load all models and aggregator
    logger.info("Initializing Master Neural Ensemble Aggregator...")
    agg = EnsembleAggregator(config=config)
    agg.load("saved_models/ensemble_aggregator")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Targeting device for PyTorch inference: {device}")
    
    temporal_model = TemporalFusionModel(name="temporal_fusion", config=config)
    temporal_model.load("saved_models/temporal_model.pt")
    
    maml_cfg = config.get("models", {}).get("meta_learner", {})
    maml_model = MAMLModel(name="maml", config={"device": device, "maml": {"inner_lr": maml_cfg.get("inner_lr", 0.01)}})
    maml_model.load("saved_models/maml_model.pt")
    
    rl_cfg = config.get("models", {}).get("rl_agent", {})
    rl_cfg["features_cols"] = features_cols
    rl_cfg["regime_cols"] = regime_cols
    rl_cfg["device"] = device
    rl_model = PPOModel(name="ppo_agent", config=rl_cfg)
    rl_model.load("saved_models/rl_agent_ppo.zip")
    
    # Wrap models
    temporal_wrapper = TemporalEnsembleWrapper(temporal_model, scaler_mean, scaler_std, raw_feature_indices, device)
    maml_wrapper = MAMLEnsembleWrapper(maml_model, scaler_mean, scaler_std, raw_feature_indices, device)
    regime_wrapper = RegimeEnsembleWrapper(regime_model, regime_mean, regime_std, hmm_features, features_df, seq_len)
    rl_wrapper = RLEnsembleWrapper(rl_model, features_cols, regime_cols, features_df, scaler_mean, scaler_std, seq_len)
    
    if fast:
        # In fast mode keep only the most critical torch models to save memory
        agg.register_model("temporal", temporal_wrapper, is_torch=True)
        agg.register_model("maml", maml_wrapper, is_torch=True)
    else:
        agg.register_model("temporal", temporal_wrapper, is_torch=True)
        agg.register_model("maml", maml_wrapper, is_torch=True)
        agg.register_model("regime", regime_wrapper, is_torch=False)
        agg.register_model("rl", rl_wrapper, is_torch=False)
    
    # Override from config
    ensemble_cfg = config.get("models", {}).get("ensemble", {})
    if "direction_threshold" in ensemble_cfg:
        agg.direction_threshold = ensemble_cfg["direction_threshold"]
        agg.signal_generator.direction_threshold = ensemble_cfg["direction_threshold"]
        
    logger.info("Warming up aggregator cache...")
    if not fast and X_master is not None:
        agg.enable_caching(X_master)
    else:
        logger.info("Fast mode: skipping aggregator cache warming.")
    
    # 6. Setup Live Trading Pipeline components
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
    
    # Broker & Execution
    broker = PaperBroker(config={"initial_capital": initial_capital})
    execution_engine = ExecutionEngine(broker=broker)
    
    tracker = PerformanceTracker(initial_capital=initial_capital)

    # God Mode Component Initialization (disabled by default for accuracy)
    if god_mode and not fast:
        try:
            synapse = DeepNeuralSynapse()
            mesh = GlobalMeshArbitrage()
            attacker = AttackerModel()
            kernel_bypass_driver = KernelBypassDriver("sfn0")
            kernel_bypass_driver.load_driver()
            logger.info("God Mode components initialized (opt-in mode)")
        except Exception as e:
            logger.warning("God Mode initialization failed, disabling", error=str(e))
            synapse = mesh = attacker = kernel_bypass_driver = None
    else:
        synapse = mesh = attacker = kernel_bypass_driver = None
        if not fast:
            logger.info("God Mode disabled (use --god-mode to enable)")

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
    
    # 7. Real-time Tick Ingestion Simulation Loop
    logger.info("Deploying Real-Time Trading Pipeline loop against live data simulator...")
    
    # Skip first sequence windows to begin trading
    start_tick = seq_len - 1
    total_ticks = len(raw_df)
    
    timestamps = raw_df.index
    closes = raw_df["close"].values
    
    for i in range(start_tick, total_ticks):
        timestamp = timestamps[i]
        close = closes[i]
        
        # Format the window input (stream windows in fast mode to save memory)
        if X_master is None:
            start = i - seq_len + 1
            window = features_arr[start: i + 1]
        else:
            window = X_master[i - seq_len + 1]
        X_input = np.expand_dims(window, axis=0)
        hour_ind = timestamp.hour / 23.0
        
        # Compute rolling volatility from recent close prices (20-bar window)
        if i >= 20:
            recent_closes = closes[i - 19 : i + 1]
            log_rets = np.diff(np.log(recent_closes))
            rolling_vol = float(np.std(log_rets))
        else:
            rolling_vol = 0.0005  # fallback for warmup period

        # Real-time market state parameters
        # CRITICAL: pip_value is the PRICE INCREMENT per pip (0.0001 for EURUSD),
        # NOT the dollar-value-per-pip. PaperBroker uses this to convert slippage
        # pips into price deltas for fill price calculation.
        market_data = {
            "close": close,
            "price": close,
            "mid_price": close,
            "spread_pips": 0.75 + np.random.rand() * 0.5, # Realistic institutional spreads
            "adv": 1000000.0,
            "pip_value": 0.0001,  # Price per pip for EURUSD (4th decimal place)
            "volatility": rolling_vol,
            # Add placeholder for God Mode features that might be used by synapse
            "usd_10y": 0.0, "vix": 0.0, "copper": 0.0, "gold": 0.0, "sp500": 0.0
        }
        
        # Feed tick update to simulated broker
        broker.update_market_state({pair: market_data})
        
        # Process tick through real models & risk gate
        pipeline.process_tick(pair, X_input, hour_ind, market_data, i - seq_len + 1)
        
        # Mark-to-market equity (cash + unrealized PnL of open positions)
        mtm_equity = broker.cash
        for pos_pair, pos_size in broker.positions.items():
            entry_px = broker.entry_prices.get(pos_pair, close)
            mtm_equity += pos_size * (close - entry_px)
        
        # Periodic equity logging
        if (i - start_tick) % 1000 == 0:
            logger.info("Real-Time Paper Account State",
                        timestamp=timestamp.isoformat(),
                        cash=broker.cash,
                        mtm_equity=round(mtm_equity, 2),
                        open_positions=broker.get_positions())
            tracker.update_equity(mtm_equity, timestamp.timestamp())
            
    # Final mark-to-market and logging open positions as final trades
    final_equity = broker.cash
    for pos_pair, pos_size in list(broker.positions.items()):
        entry_px = broker.entry_prices.get(pos_pair, closes[-1])
        realized_pnl = pos_size * (closes[-1] - entry_px)
        final_equity += realized_pnl
        
        # Log the final trade to the tracker so it counts in performance metrics
        direction = 1 if pos_size > 0 else -1
        tracker.log_trade(
            pair=pos_pair,
            direction=direction,
            size=abs(pos_size),
            pnl=realized_pnl,
            slippage_pips=0.0
        )
    
    tracker.update_equity(final_equity, timestamps[-1].timestamp())
    print("\n" + "="*80)
    print("           REAL-TIME HIGH-FIDELITY PAPER TRADING PERFORMANCE REPORT")
    print("="*80)
    print(tracker.generate_tear_sheet())
    print("="*80 + "\n")

    # Unload kernel bypass driver
    if kernel_bypass_driver:
        kernel_bypass_driver.unload_driver()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run High-Fidelity Paper Trading Simulator")
    parser.add_argument("--features", type=str, default="data/EUR_USD_features.csv", help="Path to features CSV")
    parser.add_argument("--raw", type=str, default="data/EUR_USD_ticks.csv", help="Path to raw ticks CSV")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads to use (default: CPU cores - 1)")
    parser.add_argument("--fast", action="store_true", help="Run in low-memory fast mode (reduced fidelity)")
    parser.add_argument("--disable-god-mode", action="store_true", help="Disable God Mode components (synapse, mesh, attacker)")
    args = parser.parse_args()
    
    run_real_paper_trading(features_path=args.features, raw_path=args.raw, fast=args.fast, god_mode=not args.disable_god_mode)
