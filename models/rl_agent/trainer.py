import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import pandas as pd
import structlog

from models.rl_agent.environment import ForexTradingEnv
from models.rl_agent.ppo_agent import PPOModel
from models.rl_agent.sac_agent import SACModel

logger = structlog.get_logger()


class RLTrainer:
    """
    Curriculum and Sequential Trainer for Reinforcement Learning Forex Agents.
    Supports splitting training datasets into complexity stages (e.g., low vs high volatility)
    and sequentially fitting PPO or SAC models while tracking equity progression.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "checkpoints/rl"
    ) -> None:
        self.config = config or {}
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(
            "RLTrainer initialized",
            checkpoint_dir=self.checkpoint_dir
        )

    def define_volatility_curriculum(
        self,
        df: pd.DataFrame,
        vol_col: str = "realized_volatility",
        low_vol_quantile: float = 0.5
    ) -> List[Tuple[str, pd.DataFrame]]:
        """
        Splits a DataFrame into two curriculum stages based on historical volatility:
        1. Stage 'Low Volatility': Easier environments, stable regimes.
        2. Stage 'Full Market': Complete dataset including high-stress regimes.
        """
        if vol_col not in df.columns:
            # Fallback to standard deviation of close price rolling if realized volatility not present
            df[vol_col] = df["close"].pct_change().rolling(window=20).std()
            df[vol_col] = df[vol_col].bfill()
            
        threshold = df[vol_col].quantile(low_vol_quantile)
        
        # Filter sequences
        low_vol_df = df[df[vol_col] <= threshold].copy()
        
        stages = [
            ("stage_1_low_volatility", low_vol_df),
            ("stage_2_full_market", df.copy())
        ]
        
        logger.info(
            "Curriculum defined successfully",
            vol_col=vol_col,
            threshold=threshold,
            stage_1_len=len(low_vol_df),
            stage_2_len=len(df)
        )
        
        return stages

    def train(
        self,
        agent: Union[PPOModel, SACModel],
        df: pd.DataFrame,
        features_cols: List[str],
        regime_cols: Optional[List[str]] = None,
        stages: Optional[List[Tuple[str, pd.DataFrame]]] = None,
        total_timesteps_per_stage: int = 20000,
        **kwargs: Any
    ) -> Union[PPOModel, SACModel]:
        """
        Orchestrates the sequential training loop of the RL Agent across the curriculum stages.
        """
        # Resolve stages
        if stages is None:
            # Default to dividing into low volatility first then full market
            stages = self.define_volatility_curriculum(df)
            
        logger.info(
            "Beginning sequential RL training",
            num_stages=len(stages),
            timesteps_per_stage=total_timesteps_per_stage
        )
        
        for idx, (stage_name, stage_df) in enumerate(stages):
            logger.info(
                f"Starting Curriculum Stage {idx + 1}/{len(stages)}",
                stage=stage_name,
                samples=len(stage_df)
            )
            
            # 1. Instantiate the stage environment
            action_type = "continuous" if isinstance(agent, SACModel) else "discrete"
            
            stage_env = ForexTradingEnv(
                df=stage_df,
                features_cols=features_cols,
                regime_cols=regime_cols,
                initial_balance=self.config.get("initial_balance", 10000.0),
                leverage=self.config.get("leverage", 30.0),
                multiplier=self.config.get("multiplier", 100000.0),
                slippage=self.config.get("slippage", 0.0001),
                kyle_lambda_multiplier=self.config.get("kyle_lambda_multiplier", 1.0),
                reward_config=self.config.get("reward_config"),
                force_python_fallback=self.config.get("force_python_fallback", False),
                action_space_type=action_type
            )
            
            # 2. Fit agent on the stage environment
            # Pass reset_num_timesteps=False after the first stage to preserve learning step count logs
            reset_num_timesteps = (idx == 0)
            
            agent.fit(
                X=stage_env,
                total_timesteps=total_timesteps_per_stage,
                tb_log_name=f"{agent.name}_{stage_name}",
                reset_num_timesteps=reset_num_timesteps,
                **kwargs
            )
            
            # 3. Save checkpoint for the current stage
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{agent.name}_{stage_name}.zip"
            )
            agent.save(checkpoint_path)
            
            # 4. Evaluate ending performance in stage environment
            obs, info = stage_env.reset()
            done = False
            total_reward = 0.0
            equity_path = [info["equity"]]
            
            while not done:
                action = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = stage_env.step(action)
                total_reward += reward
                equity_path.append(info["equity"])
                done = terminated or truncated
                
            logger.info(
                f"Curriculum Stage {idx + 1} completed evaluation",
                stage=stage_name,
                total_eval_reward=total_reward,
                initial_equity=equity_path[0],
                ending_equity=equity_path[-1],
                max_equity=max(equity_path),
                min_equity=min(equity_path),
                margin_called=info.get("margin_called", 0)
            )
            
        logger.info("Curriculum training successfully completed for all stages!")
        return agent
