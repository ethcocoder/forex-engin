import os
from typing import Any, Optional, Dict, Union
import pandas as pd
import numpy as np
import structlog
from stable_baselines3 import PPO

from models.base_model import BaseModel
from models.rl_agent.environment import ForexTradingEnv

logger = structlog.get_logger()


class PPOModel(BaseModel):
    """
    BaseModel-compliant wrapper around Stable-Baselines3's PPO (Proximal Policy Optimization).
    Supports training via fit() on a custom ForexTradingEnv or dataframe, and predictions.
    """

    def __init__(self, name: str = "ppo_agent", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self.config = config or {}
        
        # SB3 Model parameters
        self.policy = self.config.get("policy", "MlpPolicy")
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.n_steps = self.config.get("n_steps", 2048)
        self.batch_size = self.config.get("batch_size", 64)
        self.n_epochs = self.config.get("n_epochs", 10)
        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        self.clip_range = self.config.get("clip_range", 0.2)
        self.ent_coef = self.config.get("ent_coef", 0.0)
        self.vf_coef = self.config.get("vf_coef", 0.5)
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)
        self.device = self.config.get("device", "auto")
        
        # Internal model storage
        self.model: Optional[PPO] = None
        
        logger.info(
            "PPOModel wrapper initialized",
            policy=self.policy,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            n_steps=self.n_steps,
            device=self.device
        )

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "PPOModel":
        """
        Fits the PPO policy actor on the environment.
        
        Args:
            X: Can be a ForexTradingEnv instance or a pandas DataFrame. If DataFrame,
               an env is created internally using self.config.
            y: Ignored (standard supervised interface alignment).
            
        Keyword Args:
            total_timesteps: Total timesteps to train (default: 50,000)
            tb_log_name: Tensorboard log directory name
            callback: Stable-baselines3 callbacks
        """
        total_timesteps = kwargs.get("total_timesteps", self.config.get("total_timesteps", 50000))
        tb_log_name = kwargs.get("tb_log_name", self.name)
        callback = kwargs.get("callback", None)
        
        # 1. Resolve environment
        if isinstance(X, ForexTradingEnv):
            env = X
        elif isinstance(X, pd.DataFrame):
            features_cols = self.config.get("features_cols")
            regime_cols = self.config.get("regime_cols", [])
            if not features_cols:
                raise ValueError("Config missing 'features_cols' required to instantiate env from DataFrame.")
                
            env = ForexTradingEnv(
                df=X,
                features_cols=features_cols,
                regime_cols=regime_cols,
                initial_balance=self.config.get("initial_balance", 10000.0),
                leverage=self.config.get("leverage", 30.0),
                multiplier=self.config.get("multiplier", 100000.0),
                slippage=self.config.get("slippage", 0.0001),
                kyle_lambda_multiplier=self.config.get("kyle_lambda_multiplier", 1.0),
                reward_config=self.config.get("reward_config"),
                force_python_fallback=self.config.get("force_python_fallback", False)
            )
        else:
            raise TypeError("X must be a ForexTradingEnv or pandas DataFrame.")
            
        # 2. Instantiate or update the stable-baselines3 model
        if self.model is None:
            self.model = PPO(
                policy=self.policy,
                env=env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                device=self.device,
                tensorboard_log=self.config.get("tensorboard_log", None)
            )
        else:
            self.model.set_env(env)
            
        logger.info(
            "Starting PPOModel fit/train run",
            timesteps=total_timesteps,
            env_steps=len(env.df)
        )
        
        # 3. Learn policy
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=callback,
            reset_num_timesteps=kwargs.get("reset_num_timesteps", True)
        )
        
        logger.info("PPOModel training completed successfully")
        return self

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """
        Runs policy inference on the observation(s).
        
        Args:
            X: Observation array (single state vector or batch of states).
            
        Keyword Args:
            deterministic: Whether to return deterministic action (default: True)
            
        Returns:
            Tuple: (action(s), state(s)) or action(s) based on format
        """
        if self.model is None:
            raise RuntimeError("Cannot predict: PPOModel has not been trained (fit) or loaded yet.")
            
        deterministic = kwargs.get("deterministic", True)
        
        # SB3 predict returns (actions, states)
        action, state = self.model.predict(X, deterministic=deterministic)
        return action

    def save(self, path: str, **kwargs: Any) -> None:
        """
        Saves the underlying Stable-Baselines3 PPO model to path.
        """
        if self.model is None:
            raise RuntimeError("Cannot save: PPOModel is not initialized.")
            
        # Standard SB3 save
        self.model.save(path)
        logger.info("Saved PPOModel successfully", destination=path)

    def load(self, path: str, **kwargs: Any) -> None:
        """
        Loads a serialized Stable-Baselines3 PPO model.
        """
        # load the model using SB3's load method
        self.model = PPO.load(path, device=self.device)
        logger.info("Loaded PPOModel successfully from path", source=path)
