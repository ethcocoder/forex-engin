import numpy as np
import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from scipy.stats import ks_2samp

logger = structlog.get_logger()

class GOATAlphaDecayMonitor:
    """
    GOAT Alpha Decay & Autonomous Evolution System.
    
    Implements:
    1. Multi-window IC (Information Coefficient) monitoring.
    2. Concept Drift Detection (KS Test) on prediction distributions.
    3. Performance degradation triggers (Sharpe/Drawdown).
    4. Autonomous retraining orchestration.
    """

    def __init__(
        self,
        lookback_windows: List[int] = [100, 500, 1000, 5000],
        decay_threshold: float = 0.65,
        drift_p_value: float = 0.01,
        retrain_callback: Any = None,
        evolution_callback: Any = None
    ) -> None:
        self.lookback_windows = lookback_windows
        self.decay_threshold = decay_threshold
        self.drift_p_value = drift_p_value
        self.retrain_callback = retrain_callback
        self.evolution_callback = evolution_callback
        
        self.prediction_history: List[float] = []
        self.actual_history: List[float] = []
        self.pnl_history: List[float] = []
        self.timestamps: List[datetime] = []
        
        self.last_retrain_time = datetime.utcnow()
        self.retrain_count = 0
        
        logger.info("GOATAlphaDecayMonitor initialized", 
                    threshold=decay_threshold, 
                    drift_p=drift_p_value)

    def log_step(self, predicted_return: float, actual_return: float, pnl: float) -> None:
        """Logs a single step of model performance."""
        self.prediction_history.append(predicted_return)
        self.actual_history.append(actual_return)
        self.pnl_history.append(pnl)
        self.timestamps.append(datetime.utcnow())
        
        # Check for decay and drift periodically
        if len(self.prediction_history) % 100 == 0:
            self._evaluate_health()

    def _evaluate_health(self) -> None:
        """Comprehensive health check for the trading model."""
        if len(self.prediction_history) < max(self.lookback_windows):
            return

        # 1. Alpha Decay (IC Correlation)
        ic_values = {}
        for window in self.lookback_windows:
            preds = np.array(self.prediction_history[-window:])
            actuals = np.array(self.actual_history[-window:])
            correlation = np.corrcoef(preds, actuals)[0, 1]
            ic_values[window] = correlation

        short_term_ic = ic_values[self.lookback_windows[0]]
        long_term_ic = ic_values[self.lookback_windows[-1]]
        decay_ratio = short_term_ic / (long_term_ic + 1e-6)

        # 2. Concept Drift Detection (KS Test)
        # Compare current predictions to historical 'golden' distribution
        historical_window = 2000
        current_window = 200
        if len(self.prediction_history) > historical_window + current_window:
            hist_preds = np.array(self.prediction_history[-(historical_window+current_window):-current_window])
            curr_preds = np.array(self.prediction_history[-current_window:])
            ks_stat, p_val = ks_2samp(hist_preds, curr_preds)
            drift_detected = p_val < self.drift_p_value
        else:
            drift_detected = False
            p_val = 1.0

        # 3. Performance Metrics
        recent_pnl = self.pnl_history[-500:]
        sharpe = np.mean(recent_pnl) / (np.std(recent_pnl) + 1e-6) * np.sqrt(252)
        
        logger.info(
            "Model Health Check",
            short_ic=f"{short_term_ic:.4f}",
            decay_ratio=f"{decay_ratio:.2f}",
            drift_p=f"{p_val:.4f}",
            recent_sharpe=f"{sharpe:.2f}"
        )

        # TRIGGER LOGIC
        trigger_reason = None
        if decay_ratio < self.decay_threshold and long_term_ic > 0.03:
            trigger_reason = "ALPHA_DECAY"
        elif drift_detected:
            trigger_reason = "CONCEPT_DRIFT"
        elif sharpe < 0.5 and len(recent_pnl) >= 500:
            trigger_reason = "PERFORMANCE_DEGRADATION"

        if trigger_reason:
            self._trigger_autonomous_evolution(trigger_reason)

    def _trigger_autonomous_evolution(self, reason: str) -> None:
        """Orchestrates the autonomous retrain/evolution sequence."""
        # Prevent spamming retrains
        if datetime.utcnow() - self.last_retrain_time < timedelta(hours=4):
            logger.info("Retrain suppressed: too soon since last update", reason=reason)
            return

        logger.critical(f"AUTONOMOUS EVOLUTION TRIGGERED: {reason}")
        
        self.retrain_count += 1
        self.last_retrain_time = datetime.utcnow()
        
        # 1. Immediate Action: Fast Adaptation (MAML/Online)
        if self.evolution_callback:
            logger.info("Executing fast online adaptation...")
            self.evolution_callback()
            
        # 2. Background Action: Full Retrain
        if self.retrain_callback:
            logger.info("Queueing full model retrain on expanded dataset...")
            self.retrain_callback()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns detailed health metrics for the control suite."""
        if not self.prediction_history:
            return {"status": "WARMUP"}
            
        return {
            "uptime_since_retrain": str(datetime.utcnow() - self.last_retrain_time),
            "retrain_count": self.retrain_count,
            "current_ic": np.corrcoef(self.prediction_history[-100:], self.actual_history[-100:])[0, 1] if len(self.prediction_history) >= 100 else 0.0,
            "status": "HEALTHY" if datetime.utcnow() - self.last_retrain_time < timedelta(days=1) else "STALE"
        }
