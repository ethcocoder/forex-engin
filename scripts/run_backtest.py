import os
import sys
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import structlog
import pickle
import torch
import time

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.temporal.combined import TemporalFusionModel
from models.meta_learner.maml import MAMLModel
from models.regime.combined import RegimeEnsembleEstimator
from models.rl_agent.ppo_agent import PPOModel
from models.ensemble.aggregator import EnsembleAggregator, AlphaSignal
from configs.loader import load_config

from backtesting.portfolio import BacktestPortfolio
from backtesting.data_handler import CSVDataHandler
from backtesting.engines.vectorized import VectorizedBacktestEngine
from backtesting.engines.event_driven import EventDrivenBacktestEngine
from backtesting.performance import PerformanceCalculator
from risk.risk_engine import RiskEngine

logger = structlog.get_logger()

# -------------------------------------------------------------------------
# Dynamic PyTorch wrappers and submodel wrappers (must match train_ensemble.py)
# -------------------------------------------------------------------------
class TorchEnsembleModelWrapper(torch.nn.Module):
    def __init__(self, inner_torch_model, scaler_mean, scaler_std, raw_feature_indices, device):
        super().__init__()
        self.inner_model = inner_torch_model
        self.scaler_mean = torch.tensor(scaler_mean, dtype=torch.float32).to(device)
        self.scaler_std = torch.tensor(scaler_std, dtype=torch.float32).to(device)
        self.raw_feature_indices = raw_feature_indices
        
    def forward(self, x):
        x_raw = x[:, :, self.raw_feature_indices]
        x_scaled = (x_raw - self.scaler_mean) / self.scaler_std
        x_scaled = torch.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.inner_model(x_scaled)


class TemporalEnsembleWrapper:
    def __init__(self, temporal_model, scaler_mean, scaler_std, raw_feature_indices, device):
        self.temporal_model = temporal_model
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.raw_feature_indices = raw_feature_indices
        self.model = TorchEnsembleModelWrapper(
            temporal_model.model, scaler_mean, scaler_std, raw_feature_indices, device
        )
        
    def predict(self, X, **kwargs):
        X_raw = X[:, :, self.raw_feature_indices]
        X_scaled = (X_raw - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.temporal_model.predict(X_scaled)


class MAMLEnsembleWrapper:
    def __init__(self, maml_model, scaler_mean, scaler_std, raw_feature_indices, device):
        self.maml_model = maml_model
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.raw_feature_indices = raw_feature_indices
        self.model = TorchEnsembleModelWrapper(
            maml_model.model, scaler_mean, scaler_std, raw_feature_indices, device
        )
        
    def predict(self, X, **kwargs):
        X_raw = X[:, :, self.raw_feature_indices]
        X_scaled = (X_raw - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.maml_model.predict(X_scaled)


class RegimeEnsembleWrapper:
    def __init__(self, regime_model, regime_mean, regime_std, hmm_features, features_df, seq_len=60):
        self.regime_model = regime_model
        self.regime_mean = regime_mean
        self.regime_std = regime_std
        self.hmm_features = hmm_features
        self.features_df = features_df
        self.seq_len = seq_len
        
    def predict(self, X, **kwargs):
        # Extract regime probabilities from the last columns of X (which has been appended with regime columns)
        # Note: X is shape [n_samples, seq_len, d_feat_total], regime probabilities are at the end
        return X[:, -1, -4:]


class RLEnsembleWrapper:
    def __init__(self, ppo_model, features_cols, regime_cols, features_df, scaler_mean, scaler_std, seq_len=60):
        self.ppo_model = ppo_model
        self.features_cols = features_cols
        self.regime_cols = regime_cols
        self.features_df = features_df
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.seq_len = seq_len
        
    def predict(self, X, **kwargs):
        n_samples = X.shape[0]
        feats_raw = X[:, -1, :len(self.features_cols)]
        feats = (feats_raw - self.scaler_mean) / self.scaler_std
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pull environment observations variables from kwargs
        pos = np.full((n_samples, 1), kwargs.get("current_position", 0.0), dtype=np.float32)
        unrealized = np.full((n_samples, 1), kwargs.get("unrealized_pnl", 0.0) / 10000.0, dtype=np.float32)
        time_ind = np.full((n_samples, 1), kwargs.get("time_indicator", 0.0), dtype=np.float32)
        
        regimes = X[:, -1, -len(self.regime_cols):]
        
        obs = np.hstack([feats, pos, unrealized, time_ind, regimes])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        actions, _ = self.ppo_model.model.predict(obs, deterministic=True)
        
        action_mapping = {0: 0.0, 1: 0.5, 2: 1.0, 3: -0.5, 4: -1.0}
        pred = np.array([action_mapping[int(a)] for a in actions], dtype=np.float64)
        return pred

# -------------------------------------------------------------------------
# Custom Backtest Engine Implementations
# -------------------------------------------------------------------------
class CustomVectorizedEngine(VectorizedBacktestEngine):
    def __init__(self, data_handler, portfolio, strategy, X_valid, seq_len, config=None):
        super().__init__(data_handler, portfolio, config)
        self.strategy = strategy
        self.X_valid = X_valid
        self.seq_len = seq_len
        
    def run(self) -> Dict[str, Any]:
        logger.info("Running custom vectorized backtest with Ensemble Aggregator")
        pair = self.config.get("primary_pair", "EURUSD")
        self.data_handler.load_data()
        df = self.data_handler.data.get(pair)
        
        if df is None or df.empty:
            logger.error("No data available for vectorized run", pair=pair)
            return {"status": "no_data"}
            
        n_samples = len(self.X_valid)
        
        logger.info("Computing signals in batch...")
        # To avoid overheads, run predict in batches of 1000
        batch_size = 1000
        predictions = []
        for i in range(0, n_samples, batch_size):
            X_batch = self.X_valid[i : i + batch_size]
            # Call predict without signal wrapper (returns float returns predictions)
            preds = [self.strategy.predict(X_batch[j], return_signal=False) for j in range(len(X_batch))]
            predictions.extend(preds)
            
        predictions = np.array(predictions)
        
        # Align prediction outputs with the close prices
        # X_valid corresponds to sliding windows starting from seq_len - 1 to len(df) - horizon
        # Let's align df columns
        df_sub = df.iloc[self.seq_len - 1 : self.seq_len - 1 + n_samples].copy()
        df_sub["market_returns"] = df_sub["close"].pct_change()
        
        # Signal direction mapping: Long (+1) if prediction > threshold, Short (-1) if prediction < -threshold, else Flat (0)
        threshold = self.strategy.direction_threshold
        df_sub["signal"] = np.where(predictions > threshold, 1, np.where(predictions < -threshold, -1, 0))
        
        # Shift signal by 1 step to avoid lookahead bias
        df_sub["strategy_returns"] = df_sub["signal"].shift(1) * df_sub["market_returns"]
        
        df_sub["cum_returns"] = (1 + df_sub["strategy_returns"].fillna(0)).cumprod()
        equity_curve = self.portfolio.initial_capital * df_sub["cum_returns"]
        
        # Realized trades
        df_sub["trade_trigger"] = df_sub["signal"].diff().fillna(0)
        trades_triggered = df_sub[df_sub["trade_trigger"] != 0]
        
        raw_trades = []
        for ts, row in trades_triggered.iterrows():
            raw_trades.append({
                "pair": pair,
                "direction": int(row["signal"]),
                "size": 10000.0,
                "pnl": float(row["strategy_returns"] * self.portfolio.initial_capital),
                "timestamp": ts
            })
            
        self.portfolio.equity_history = equity_curve.tolist()
        
        perf_metrics = PerformanceCalculator.calculate_metrics(
            self.portfolio.equity_history,
            returns=df_sub["strategy_returns"].dropna().tolist()
        )
        trade_metrics = PerformanceCalculator.calculate_trade_metrics(raw_trades)
        
        logger.info("Custom vectorized backtest complete", final_equity=self.portfolio.equity_history[-1])
        return {
            "performance": perf_metrics,
            "trades": trade_metrics,
            "final_equity": self.portfolio.equity_history[-1],
            "raw_trades": raw_trades
        }


class CustomEventDrivenEngine(EventDrivenBacktestEngine):
    def __init__(self, data_handler, portfolio, risk_engine, strategy, X_master, seq_len, features_df, config=None):
        super().__init__(data_handler, portfolio, risk_engine, strategy, config)
        self.X_master = X_master
        self.seq_len = seq_len
        self.features_df = features_df
        self.current_bar_index = 0
        
    def _handle_market(self, bar: Dict[str, Any]) -> None:
        pair = bar["pair"]
        
        # Ensure we have enough history to construct a sequence window
        if self.current_bar_index < self.seq_len - 1:
            self.current_bar_index += 1
            return
            
        # Get current sliding window
        window = self.X_master[self.current_bar_index - self.seq_len + 1]
        X_input = np.expand_dims(window, axis=0)
        
        # Calculate time indicator (hour / 23.0)
        timestamp = bar["timestamp"]
        hour_ind = timestamp.hour / 23.0
        
        # Pass environment parameters to RL model wrapper
        current_pos = self.portfolio.positions.get(pair, 0.0)
        unrealized = 0.0
        if current_pos != 0.0:
            entry = self.portfolio.avg_entry.get(pair, bar["close"])
            unrealized = current_pos * (bar["close"] - entry)
            
        # Predict AlphaSignal
        signal = self.strategy.predict(
            X_input,
            return_signal=True,
            current_position=current_pos,
            unrealized_pnl=unrealized,
            time_indicator=hour_ind
        )
        
        if signal.direction != 0:
            self.events_queue.put({
                "type": "SIGNAL",
                "pair": pair,
                "signal": signal,
                "bar": bar
            })
            
        self.current_bar_index += 1


def main():
    parser = argparse.ArgumentParser(description="Run full backtesting simulation using Ensemble Stacker")
    parser.add_argument("--features", type=str, default="data/EUR_USD_features.csv", help="Path to features CSV")
    parser.add_argument("--raw", type=str, default="data/EUR_USD_ticks.csv", help="Path to raw ticks CSV")
    parser.add_argument("--ensemble", type=str, default="saved_models/ensemble_aggregator", help="Path to saved ensemble weights")
    parser.add_argument("--seq_len", type=int, default=60, help="Lookback sequence length")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--mode", type=str, default="both", choices=["vectorized", "event_driven", "both"], help="Backtesting mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.features) or not os.path.exists(args.raw):
        logger.error("Features or raw data files not found. Ensure features are generated.")
        sys.exit(1)
        
    logger.info("Loading DataFrames...")
    features_df = pd.read_csv(args.features, index_col="timestamp", parse_dates=True)
    raw_df = pd.read_csv(args.raw, index_col="timestamp", parse_dates=True)
    
    # Align indexes
    common_idx = raw_df.index.intersection(features_df.index)
    raw_df = raw_df.loc[common_idx]
    features_df = features_df.loc[common_idx]
    
    logger.info("Loading configuration...")
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
        app_config = load_config(config_path)
        config = app_config.model_dump() if hasattr(app_config, "model_dump") else app_config.dict()
    except Exception as e:
        logger.warning(f"Could not load config file, using empty dict: {e}")
        config = {}
        
    # Load Regime Ensemble
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
    windows = sliding_window_view(regime_features_scaled, window_shape=(args.seq_len, regime_features_scaled.shape[1]))
    X_regime = windows.squeeze(1)
    probs = regime_model.predict(X_regime, return_proba=True)
    
    # Pad probs to match features_df length
    padding = np.tile(probs[0], (args.seq_len - 1, 1))
    aligned_probs = np.vstack([padding, probs])
    
    regime_cols = [f"regime_{i}" for i in range(probs.shape[1])]
    for i, col in enumerate(regime_cols):
        features_df[col] = aligned_probs[:, i]
        
    exclude = ["timestamp", "close", "open", "high", "low", "volume", "bid", "ask"] + regime_cols
    features_cols = [col for col in features_df.columns if col not in exclude]
    
    # Load feature scalers
    with open("saved_models/feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        scaler_mean = scaler["mean"]
        scaler_std = scaler["std"]
        
    raw_feature_indices = [features_df.columns.get_loc(col) for col in features_cols]
    
    # We keep the master features UN-SCALED because individual wrappers handle their own scaling
    features_arr = features_df.copy().values
    features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Generate windows
    windows = sliding_window_view(features_arr, window_shape=(args.seq_len, features_arr.shape[1]))
    X_master = np.copy(windows.squeeze(1))
    
    n_samples = len(X_master) - args.horizon
    X_valid = np.copy(X_master[:n_samples])
    
    # Load all models and aggregator
    logger.info("Loading EnsembleAggregator and registered submodels...")
    agg = EnsembleAggregator(config=config)
    agg.load(args.ensemble)
    
    temporal_model = TemporalFusionModel(name="temporal_fusion", config=config)
    temporal_model.load("saved_models/temporal_model.pt")
    
    maml_cfg = config.get("models", {}).get("meta_learner", {})
    maml_model = MAMLModel(name="maml", config={"device": args.device, "maml": {"inner_lr": maml_cfg.get("inner_lr", 0.01)}})
    maml_model.load("saved_models/maml_model.pt")
    
    rl_cfg = config.get("models", {}).get("rl_agent", {})
    rl_cfg["features_cols"] = features_cols
    rl_cfg["regime_cols"] = regime_cols
    rl_cfg["device"] = args.device
    rl_model = PPOModel(name="ppo_agent", config=rl_cfg)
    rl_model.load("saved_models/rl_agent_ppo.zip")
    
    # Wrap models
    temporal_wrapper = TemporalEnsembleWrapper(temporal_model, scaler_mean, scaler_std, raw_feature_indices, args.device)
    maml_wrapper = MAMLEnsembleWrapper(maml_model, scaler_mean, scaler_std, raw_feature_indices, args.device)
    regime_wrapper = RegimeEnsembleWrapper(regime_model, regime_mean, regime_std, hmm_features, features_df, args.seq_len)
    rl_wrapper = RLEnsembleWrapper(rl_model, features_cols, regime_cols, features_df, scaler_mean, scaler_std, args.seq_len)
    
    agg.register_model("temporal", temporal_wrapper, is_torch=True)
    agg.register_model("maml", maml_wrapper, is_torch=True)
    agg.register_model("regime", regime_wrapper, is_torch=False)
    agg.register_model("rl", rl_wrapper, is_torch=False)
    
    # -------------------------------------------------------------------------
    # Execute Backtests
    # -------------------------------------------------------------------------
    results = {}
    
    if args.mode in ["vectorized", "both"]:
        logger.info("Executing Vectorized Backtest...")
        # Prepare CSV handler
        # Since CSVDataHandler expects a pair.csv in csv_dir, we temporarily copy EUR_USD_ticks.csv to EURUSD.csv
        # under the 'data' directory
        pair = "EURUSD"
        pair_csv = os.path.join("data", f"{pair}.csv")
        if not os.path.exists(pair_csv):
            import shutil
            shutil.copy(args.raw, pair_csv)
            
        data_handler = CSVDataHandler(csv_dir="data", pairs=[pair])
        portfolio = BacktestPortfolio(initial_capital=10000.0)
        
        engine = CustomVectorizedEngine(
            data_handler=data_handler,
            portfolio=portfolio,
            strategy=agg,
            X_valid=X_valid,
            seq_len=args.seq_len,
            config={"primary_pair": pair}
        )
        results["vectorized"] = engine.run()
        
    if args.mode in ["event_driven", "both"]:
        logger.info("Executing Event-Driven Backtest...")
        pair = "EURUSD"
        pair_csv = os.path.join("data", f"{pair}.csv")
        if not os.path.exists(pair_csv):
            import shutil
            shutil.copy(args.raw, pair_csv)
            
        data_handler = CSVDataHandler(csv_dir="data", pairs=[pair])
        portfolio = BacktestPortfolio(initial_capital=10000.0)
        risk_engine = RiskEngine(config=config.get("risk", {}))
        
        engine = CustomEventDrivenEngine(
            data_handler=data_handler,
            portfolio=portfolio,
            risk_engine=risk_engine,
            strategy=agg,
            X_master=X_master,
            seq_len=args.seq_len,
            features_df=features_df,
            config={"primary_pair": pair}
        )
        results["event_driven"] = engine.run()
        
    # -------------------------------------------------------------------------
    # Print Backtest Comparison Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("                    FOREX NEURAL ENGINE BACKTEST SUMMARY")
    print("="*80)
    
    headers = ["Metric", "Vectorized Engine", "Event-Driven Engine"]
    print(f"{headers[0]:<35} | {headers[1]:<20} | {headers[2]:<20}")
    print("-"*80)
    
    metrics_to_show = [
        ("total_return_pct", "Total Return (%)", "{:.2f}%"),
        ("max_drawdown_pct", "Max Drawdown (%)", "{:.2f}%"),
        ("annualized_sharpe", "Sharpe Ratio", "{:.3f}"),
        ("annualized_sortino", "Sortino Ratio", "{:.3f}"),
    ]
    
    for key, label, fmt in metrics_to_show:
        vec_val = "N/A"
        evt_val = "N/A"
        if "vectorized" in results and "performance" in results["vectorized"]:
            vec_val = fmt.format(results["vectorized"]["performance"].get(key, 0.0))
        if "event_driven" in results and "performance" in results["event_driven"]:
            evt_val = fmt.format(results["event_driven"]["performance"].get(key, 0.0))
        print(f"{label:<35} | {vec_val:<20} | {evt_val:<20}")
        
    trade_metrics = [
        ("total_trades", "Total Trades", "{}"),
        ("win_rate_pct", "Win Rate (%)", "{:.2f}%"),
        ("profit_factor", "Profit Factor", "{:.3f}"),
    ]
    
    print("-"*80)
    for key, label, fmt in trade_metrics:
        vec_val = "N/A"
        evt_val = "N/A"
        if "vectorized" in results and "trades" in results["vectorized"]:
            vec_val = fmt.format(results["vectorized"]["trades"].get(key, 0.0))
        if "event_driven" in results and "trades" in results["event_driven"]:
            evt_val = fmt.format(results["event_driven"]["trades"].get(key, 0.0))
        print(f"{label:<35} | {vec_val:<20} | {evt_val:<20}")
        
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
