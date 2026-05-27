import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
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

logger = structlog.get_logger()


class RealTimePipeline(TradingPipeline):
    """
    Extends TradingPipeline to perform automated exits based on dynamic holding decay steps.
    """
    def __init__(self, ensemble, risk_engine, execution_engine, initial_capital=100000.0):
        super().__init__(ensemble, risk_engine, execution_engine, initial_capital=initial_capital)
        self.trade_durations = {} # pair -> count of bars held
        self.max_hold_steps = 9   # Default signal decay steps
        
    def process_tick(self, pair: str, X_window: np.ndarray, hour_ind: float, market_data: dict, current_bar_idx: int) -> None:
        # 1. Sync Portfolio State with broker
        actual_positions = self.execution_engine.sync_portfolio_state()
        self.portfolio_state.open_positions = actual_positions

        current_pos = actual_positions.get(pair, 0.0)

        # 2. Dynamic holding & exit processing
        if current_pos != 0.0:
            self.trade_durations[pair] = self.trade_durations.get(pair, 0) + 1
            if self.trade_durations[pair] >= self.max_hold_steps:
                # Signal holding period complete -> execute market exit
                logger.info("Signal decay threshold reached. Initiating exit order.", pair=pair, steps=self.trade_durations[pair])
                direction = -1 if current_pos > 0 else 1
                from risk.risk_engine import OrderRequest
                exit_order = OrderRequest(pair=pair, direction=direction, size=abs(current_pos))
                self.execution_engine.execute(exit_order)
                self.trade_durations[pair] = 0
                return
        else:
            self.trade_durations[pair] = 0

        # Calculate unrealized pnl for RL environment wrapper state tracking
        unrealized = 0.0
        if current_pos != 0.0:
            broker = self.execution_engine.broker
            entry = getattr(broker, "entry_prices", {}).get(pair, getattr(broker, "avg_entry", {}).get(pair, market_data["close"]))
            unrealized = current_pos * (market_data["close"] - entry)

        # 3. Predict AlphaSignal using our real master EnsembleAggregator!
        signal = self.ensemble.predict(
            X_window,
            return_signal=True,
            current_position=current_pos,
            unrealized_pnl=unrealized,
            time_indicator=hour_ind,
            sample_idx=current_bar_idx
        )
        
        if signal.direction == 0:
            return  # No entry action

        # 4. Gate signal through Risk Engine
        order = self.risk_engine.gate(signal, pair, self.portfolio_state, market_data)
        
        # 5. Execute Order
        if order is not None:
            success = self.execution_engine.execute(order)
            if success:
                logger.info("Real-Time order executed successfully", pair=pair, direction=order.direction, size=order.size)
                self.trade_durations[pair] = 0
                self.max_hold_steps = signal.expected_decay_steps


def run_real_paper_trading(features_path="data/EUR_USD_features.csv", raw_path="data/EUR_USD_ticks.csv"):
    logger.info("Starting High-Fidelity Real Paper Trading Simulator...")
    
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
    features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    windows = sliding_window_view(features_arr, window_shape=(seq_len, features_arr.shape[1]))
    X_master = np.copy(windows.squeeze(1))
    
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
    agg.enable_caching(X_master)
    
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
    
    pipeline = RealTimePipeline(
        ensemble=agg,
        risk_engine=risk_engine,
        execution_engine=execution_engine,
        initial_capital=initial_capital
    )
    
    tracker = PerformanceTracker(initial_capital=initial_capital)
    
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
        
        # Format the window input
        window = X_master[i - seq_len + 1]
        X_input = np.expand_dims(window, axis=0)
        hour_ind = timestamp.hour / 23.0
        
        # Real-time market state parameters
        market_data = {
            "close": close,
            "mid_price": close,
            "spread_pips": 0.75 + np.random.rand() * 0.5, # Realistic institutional spreads
            "adv": 1000000.0,
            "pip_value": 10.0,
            "volatility": 0.0005
        }
        
        # Feed tick update to simulated broker
        broker.update_market_state({pair: market_data})
        
        # Process tick through real models & risk gate
        pipeline.process_tick(pair, X_input, hour_ind, market_data, i - seq_len + 1)
        
        # Periodic equity logging
        if (i - start_tick) % 1000 == 0:
            logger.info("Real-Time Paper Account State", timestamp=timestamp.isoformat(), equity=broker.cash, open_positions=broker.get_positions())
            tracker.update_equity(broker.cash, timestamp.timestamp())
            
    # Compile performance sheet
    tracker.update_equity(broker.cash, timestamps[-1].timestamp())
    print("\n" + "="*80)
    print("           REAL-TIME HIGH-FIDELITY PAPER TRADING PERFORMANCE REPORT")
    print("="*80)
    print(tracker.generate_tear_sheet())
    print("="*80 + "\n")


if __name__ == "__main__":
    run_real_paper_trading()
