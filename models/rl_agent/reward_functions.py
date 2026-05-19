import numpy as np
import structlog
from typing import Dict, Any

logger = structlog.get_logger()

class ForexRewardEngine:
    """
    Advanced Risk-Adjusted Reward Engine for Reinforcement Learning Forex Agent.
    
    Supports:
    1. Sortino-style downside variance penalization.
    2. VPIN-based toxicity holding penalty.
    3. Excessive churn (transaction/turnover) penalty.
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        self.config = config or {}
        
        # Hyperparameters
        self.downside_alpha = self.config.get("downside_alpha", 0.05)       # Decay factor for running downside variance
        self.downside_penalty = self.config.get("downside_penalty", 2.0)     # Coefficient for downside standard deviation penalty
        self.vpin_penalty_coef = self.config.get("vpin_penalty_coef", 0.5)  # Coefficient for toxic VPIN holdings penalty
        self.churn_penalty_coef = self.config.get("churn_penalty_coef", 0.1)# Coefficient for transaction churn penalty
        
        # State variables
        self.downside_variance = 0.0
        
        logger.info(
            "ForexRewardEngine initialized",
            downside_alpha=self.downside_alpha,
            downside_penalty=self.downside_penalty,
            vpin_penalty_coef=self.vpin_penalty_coef,
            churn_penalty_coef=self.churn_penalty_coef
        )
        
    def reset(self) -> None:
        """Resets the running risk metrics."""
        self.downside_variance = 0.0
        
    def calculate_reward(
        self,
        step_return: float,
        current_position: float,
        prev_position: float,
        vpin: float,
        info: Dict[str, Any]
    ) -> float:
        """
        Computes the complete multi-component risk-adjusted reward for the current step.
        
        Args:
            step_return: The percentage change in equity for the current step (e.g. (eq_t - eq_t-1)/eq_t-1)
            current_position: The newly taken position weight [-1.0, 1.0]
            prev_position: The previous position weight [-1.0, 1.0]
            vpin: Volume-Synchronized Probability of Toxicity (0.0 to 1.0)
            info: Dictionary containing step metrics (e.g., realized/unrealized PnL, margin called)
            
        Returns:
            The net reward to be returned to the RL step.
        """
        # 1. Base return
        base_reward = step_return
        
        # 2. Sortino downside variance calculation
        # We only penalize returns that are negative.
        if step_return < 0.0:
            # Running exponential moving variance of negative returns
            self.downside_variance = (
                self.downside_alpha * (step_return ** 2) + 
                (1.0 - self.downside_alpha) * self.downside_variance
            )
        else:
            # Decays downside variance slightly towards zero if positive return
            self.downside_variance = (1.0 - self.downside_alpha) * self.downside_variance
            
        downside_std = np.sqrt(self.downside_variance)
        downside_risk_penalty = self.downside_penalty * downside_std
        
        # 3. Holding Toxicity Penalty (VPIN-driven)
        # Penalizes holding a position when the flow toxicity (VPIN) is high.
        holding_toxicity_penalty = self.vpin_penalty_coef * vpin * abs(current_position)
        
        # 4. Excessive Churn Penalty
        # Penalizes changing position size frequently (turnover)
        churn = abs(current_position - prev_position)
        churn_penalty = self.churn_penalty_coef * churn
        
        # 5. Strict Margin Call Penalty
        margin_call_penalty = 0.0
        if info.get("margin_called", 0) == 1:
            margin_call_penalty = 1.0  # Massive penalty to discourage blowing up the account
            
        # Combine rewards
        net_reward = base_reward - downside_risk_penalty - holding_toxicity_penalty - churn_penalty - margin_call_penalty
        
        # Update info dict for transparency/debugging
        info["reward_base"] = base_reward
        info["reward_downside_penalty"] = downside_risk_penalty
        info["reward_toxicity_penalty"] = holding_toxicity_penalty
        info["reward_churn_penalty"] = churn_penalty
        info["reward_margin_call_penalty"] = margin_call_penalty
        info["downside_std"] = downside_std
        info["net_reward"] = net_reward
        
        return float(net_reward)
