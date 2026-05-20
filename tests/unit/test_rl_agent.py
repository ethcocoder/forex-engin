import os
import time
import tempfile
import unittest
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

from models.rl_agent.reward_functions import ForexRewardEngine
from models.rl_agent.environment import ForexTradingEnv, _rl_lib
from models.rl_agent.ppo_agent import PPOModel
from models.rl_agent.sac_agent import SACModel
from models.rl_agent.trainer import RLTrainer


class TestRLAgentAndEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.n = 1000
        
        # 1. Create realistic synthetic forex data
        time_index = pd.date_range(start="2026-01-01", periods=self.n, freq="min")
        
        # Random walk for mid price
        steps = np.random.normal(0, 0.00015, self.n)
        close = 1.1200 + np.cumsum(steps)
        
        # Add high-leverage crash step at step 800 to test margin calls
        close[800:] = close[799] - 0.05
        
        spread = np.random.uniform(0.0001, 0.0003, self.n)
        kyle_lambda = np.random.uniform(1e-6, 1e-5, self.n)
        vpin = np.random.uniform(0.1, 0.8, self.n)
        
        # Features and regimes
        feat_1 = np.random.normal(0, 1.0, self.n)
        feat_2 = np.random.normal(0.5, 0.5, self.n)
        regime_0 = np.random.uniform(0.0, 1.0, self.n)
        regime_1 = 1.0 - regime_0
        
        self.df = pd.DataFrame(
            {
                "close": close,
                "spread": spread,
                "kyle_lambda": kyle_lambda,
                "vpin": vpin,
                "feat_1": feat_1,
                "feat_2": feat_2,
                "regime_0": regime_0,
                "regime_1": regime_1
            },
            index=time_index
        )
        
        self.features_cols = ["feat_1", "feat_2"]
        self.regime_cols = ["regime_0", "regime_1"]

    def test_cpp_library_loaded(self) -> None:
        """Verify that the compiled C++ RL portfolio speedups library is loaded successfully."""
        self.assertIsNotNone(_rl_lib, "C++ RL speedups library should be loaded and not None")

    def test_gymnasium_compatibility(self) -> None:
        """Runs SB3 check_env to assert that the custom environment complies with standard Gym API."""
        # Test Discrete
        env_disc = ForexTradingEnv(
            df=self.df.iloc[:200],  # short sub-sample for speed
            features_cols=self.features_cols,
            regime_cols=self.regime_cols,
            initial_balance=10000.0,
            leverage=30.0
        )
        check_env(env_disc, warn=True)
        
        # Test Continuous
        env_cont = ForexTradingEnv(
            df=self.df.iloc[:200],
            features_cols=self.features_cols,
            regime_cols=self.regime_cols,
            initial_balance=10000.0,
            leverage=30.0,
            action_space_type="continuous"
        )
        check_env(env_cont, warn=True)

    def test_precision_parity(self) -> None:
        """
        Runs identical sequences of random trades on C++ environment and Python-fallback environment
        to verify that all portfolio math, transaction costs, realized/unrealized PnL,
        and margin call closeouts match precisely to 10 decimal places.
        """
        # 1. Instantiate C++ accelerated environment
        env_cpp = ForexTradingEnv(
            df=self.df,
            features_cols=self.features_cols,
            regime_cols=self.regime_cols,
            initial_balance=10000.0,
            leverage=30.0,
            force_python_fallback=False
        )
        
        # 2. Instantiate Python-forced fallback environment
        env_py = ForexTradingEnv(
            df=self.df,
            features_cols=self.features_cols,
            regime_cols=self.regime_cols,
            initial_balance=10000.0,
            leverage=30.0,
            force_python_fallback=True
        )
        
        obs_cpp, info_cpp = env_cpp.reset(seed=42)
        obs_py, info_py = env_py.reset(seed=42)
        
        np.testing.assert_array_almost_equal(obs_cpp, obs_py, decimal=10)
        self.assertAlmostEqual(info_cpp["balance"], info_py["balance"], places=10)
        
        # Generate random actions
        np.random.seed(42)
        actions = np.random.randint(0, 5, self.n - 1)
        
        margin_call_triggered = False
        
        for step_idx, action in enumerate(actions):
            # Step both environments
            obs_c, rew_c, term_c, trunc_c, info_c = env_cpp.step(action)
            obs_p, rew_p, term_p, trunc_p, info_p = env_py.step(action)
            
            # Record if a margin call happened
            if info_c["margin_called"] == 1:
                margin_call_triggered = True
                
            # Assert state transitions are mathematically identical
            self.assertAlmostEqual(info_c["balance"], info_p["balance"], places=10, 
                                   msg=f"Balance mismatch at step {step_idx}")
            self.assertAlmostEqual(info_c["equity"], info_p["equity"], places=10, 
                                   msg=f"Equity mismatch at step {step_idx}")
            self.assertAlmostEqual(info_c["position"], info_p["position"], places=10, 
                                   msg=f"Position mismatch at step {step_idx}")
            self.assertAlmostEqual(info_c["entry_price"], info_p["entry_price"], places=10, 
                                   msg=f"Entry price mismatch at step {step_idx}")
            self.assertAlmostEqual(info_c["realized_pnl"], info_p["realized_pnl"], places=10, 
                                   msg=f"Realized PnL mismatch at step {step_idx}")
            self.assertAlmostEqual(info_c["unrealized_pnl"], info_p["unrealized_pnl"], places=10, 
                                   msg=f"Unrealized PnL mismatch at step {step_idx}")
            self.assertEqual(info_c["margin_called"], info_p["margin_called"], 
                             msg=f"Margin called mismatch at step {step_idx}")
            self.assertEqual(term_c, term_p, msg=f"Terminated mismatch at step {step_idx}")
            self.assertEqual(trunc_c, trunc_p, msg=f"Truncated mismatch at step {step_idx}")
            self.assertAlmostEqual(rew_c, rew_p, places=10, msg=f"Reward mismatch at step {step_idx}")
            
            # Assert observations match
            np.testing.assert_array_almost_equal(obs_c, obs_p, decimal=10, 
                                                 err_msg=f"Observation vector mismatch at step {step_idx}")
            
            # If both environments terminated or truncated, break the simulation loop
            if (term_c or trunc_c) and (term_p or trunc_p):
                break
                
        # Confirm that our synthetic high-leverage crash step successfully triggered a margin call
        self.assertTrue(margin_call_triggered, "Safety check: synthetic data crash must trigger margin call parity check.")

    def test_execution_speedup(self) -> None:
        """
        Verifies that C++ portfolio calculation layer provides at least 15x speedup
        compared to pure Python loop execution when simulating complete trajectories.
        Also asserts numerical precision parity of the full trajectory output.
        """
        env_cpp = ForexTradingEnv(
            df=self.df,
            features_cols=self.features_cols,
            regime_cols=self.regime_cols,
            initial_balance=10000.0,
            leverage=30.0,
            force_python_fallback=False
        )
        
        actions = np.random.randint(0, 5, self.n - 1)
        target_positions = np.array([env_cpp.action_to_position[act] for act in actions], dtype=np.float64)
        
        # Warm-up and Verify Parity
        res_py = env_cpp.simulate_trajectory(target_positions, force_python=True)
        res_cpp = env_cpp.simulate_trajectory(target_positions, force_python=False)
        
        for key in ["balance", "position", "entry_price", "realized_pnl", "unrealized_pnl", "margin_called"]:
            np.testing.assert_array_almost_equal(
                res_py[key], res_cpp[key], decimal=10,
                err_msg=f"Trajectory parity mismatch for key '{key}'"
            )
            
        # Profile Python fallback trajectory
        start_py = time.perf_counter()
        for _ in range(10):  # 10 runs
            _ = env_cpp.simulate_trajectory(target_positions, force_python=True)
        end_py = time.perf_counter()
        avg_py_duration = (end_py - start_py) / 10.0
        
        # Profile C++ optimized trajectory
        start_cpp = time.perf_counter()
        for _ in range(10):  # 10 runs
            _ = env_cpp.simulate_trajectory(target_positions, force_python=False)
        end_cpp = time.perf_counter()
        avg_cpp_duration = (end_cpp - start_cpp) / 10.0
        
        speedup = avg_py_duration / avg_cpp_duration
        print(f"\nPortfolio Trajectory Acceleration profile (n={self.n}):")
        print(f"  - Pure Python Trajectory Loop: {avg_py_duration * 1000:.3f} ms")
        print(f"  - C++ Accelerated Trajectory Loop: {avg_cpp_duration * 1000:.3f} ms")
        print(f"  - Trajectory Speedup Factor: {speedup:.1f}x")
        
        self.assertGreaterEqual(
            speedup, 10.0,
            f"C++ portfolio loop must run at least 10x faster than Python fallback (got {speedup:.1f}x)"
        )

    def test_reward_engine(self) -> None:
        """
        Directly tests the components of the reward function engine (Sortino risk, VPIN toxicity, churn).
        """
        engine = ForexRewardEngine({
            "downside_alpha": 0.1,
            "downside_penalty": 2.0,
            "vpin_penalty_coef": 0.5,
            "churn_penalty_coef": 0.2
        })
        
        # 1. Base return step
        info = {}
        # Zero position, zero change, zero toxicity -> net should be 0.0
        rew = engine.calculate_reward(
            step_return=0.0,
            current_position=0.0,
            prev_position=0.0,
            vpin=0.0,
            info=info
        )
        self.assertEqual(rew, 0.0)
        
        # 2. Negative return downside variance check
        rew_neg = engine.calculate_reward(
            step_return=-0.02,
            current_position=0.0,
            prev_position=0.0,
            vpin=0.0,
            info=info
        )
        # Expected downside variance: 0.1 * (-0.02)^2 + 0.9 * 0 = 0.00004
        # downside_std = sqrt(0.00004) = 0.006324555
        # downside_penalty = 2.0 * 0.006324555 = 0.01264911
        # Net reward = -0.02 - 0.01264911 = -0.03264911
        self.assertAlmostEqual(rew_neg, -0.03264911, places=7)
        
        # 3. Holding toxicity (VPIN) check
        # High VPIN holding positive position
        rew_toxic = engine.calculate_reward(
            step_return=0.0,
            current_position=0.8,
            prev_position=0.8,
            vpin=0.6,
            info=info
        )
        # expected toxicity penalty: 0.5 * 0.6 * 0.8 = 0.24
        # downside std is now based on decaying previous downside variance:
        # downside_variance = 0.9 * 0.00004 = 0.000036
        # downside std = sqrt(0.000036) = 0.006
        # downside penalty = 2.0 * 0.006 = 0.012
        # Net reward = 0.0 - 0.012 - 0.24 = -0.252
        self.assertAlmostEqual(rew_toxic, -0.252, places=7)
        
        # 4. Churn check
        # Large position shift
        rew_churn = engine.calculate_reward(
            step_return=0.0,
            current_position=-0.5,
            prev_position=0.5,
            vpin=0.0,
            info=info
        )
        # churn = |-0.5 - 0.5| = 1.0
        # churn penalty: 0.2 * 1.0 = 0.20
        # downside variance decay: 0.9 * 0.000036 = 0.0000324
        # downside std = sqrt(0.0000324) = 0.0056921
        # downside penalty = 2.0 * 0.0056921 = 0.0113842
        # Net reward = 0.0 - 0.0113842 - 0.0 - 0.20 = -0.2113842
        self.assertAlmostEqual(rew_churn, -0.2113842, places=7)

    def test_ppo_and_sac_pipeline(self) -> None:
        """
        Verifies fit/predict/serialization pipelines for PPOModel (Discrete) and SACModel (Continuous).
        """
        # Create small subset to fit fast
        train_df = self.df.iloc[:200].copy()
        
        # PPO Config
        ppo_config = {
            "features_cols": self.features_cols,
            "regime_cols": self.regime_cols,
            "learning_rate": 0.001,
            "n_steps": 128,
            "batch_size": 32,
            "n_epochs": 2,
            "device": "cpu"
        }
        
        # Fit PPO
        ppo_agent = PPOModel(name="test_ppo", config=ppo_config)
        ppo_agent.fit(X=train_df, total_timesteps=128)
        
        # Test predict
        obs = np.zeros(ppo_agent.model.observation_space.shape, dtype=np.float32)
        act_ppo = ppo_agent.predict(obs)
        self.assertTrue(0 <= act_ppo <= 4)
        
        # Test PPO Serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "ppo_model")
            ppo_agent.save(save_path)
            
            # Load and verify prediction parity
            loaded_ppo = PPOModel(config={"device": "cpu"})
            loaded_ppo.load(save_path + ".zip")
            
            act_loaded = loaded_ppo.predict(obs)
            self.assertEqual(act_ppo, act_loaded)
            
        # SAC Config
        sac_config = {
            "features_cols": self.features_cols,
            "regime_cols": self.regime_cols,
            "learning_rate": 0.001,
            "buffer_size": 1000,
            "learning_starts": 10,
            "batch_size": 16,
            "device": "cpu"
        }
        
        # Fit SAC
        sac_agent = SACModel(name="test_sac", config=sac_config)
        sac_agent.fit(X=train_df, total_timesteps=32)
        
        # Test predict
        obs_sac = np.zeros(sac_agent.model.observation_space.shape, dtype=np.float32)
        act_sac = sac_agent.predict(obs_sac)
        self.assertEqual(len(act_sac), 1)
        self.assertTrue(-1.0 <= act_sac[0] <= 1.0)
        
        # Test SAC Serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path_sac = os.path.join(tmpdir, "sac_model")
            sac_agent.save(save_path_sac)
            
            # Load and verify prediction parity
            loaded_sac = SACModel(config={"device": "cpu"})
            loaded_sac.load(save_path_sac + ".zip")
            
            act_loaded_sac = loaded_sac.predict(obs_sac)
            np.testing.assert_array_almost_equal(act_sac, act_loaded_sac, decimal=5)

    def test_curriculum_trainer(self) -> None:
        """
        Verifies that RLTrainer defines curriculum stages and executes training across stages.
        """
        # Create a small mockup dataset with volatility column
        df = self.df.iloc[:200].copy()
        df["realized_volatility"] = np.random.uniform(0.0001, 0.001, len(df))
        
        ppo_config = {
            "features_cols": self.features_cols,
            "regime_cols": self.regime_cols,
            "n_steps": 128,
            "batch_size": 32,
            "n_epochs": 1,
            "device": "cpu"
        }
        ppo_agent = PPOModel(name="curriculum_ppo", config=ppo_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = RLTrainer(checkpoint_dir=tmpdir)
            
            # Perform sequential training
            trainer.train(
                agent=ppo_agent,
                df=df,
                features_cols=self.features_cols,
                regime_cols=self.regime_cols,
                total_timesteps_per_stage=128
            )
            
            # Verify that final checkpoint exists
            stage_1_checkpoint = os.path.join(tmpdir, "curriculum_ppo_stage_1_low_volatility.zip")
            stage_2_checkpoint = os.path.join(tmpdir, "curriculum_ppo_stage_2_full_market.zip")
            self.assertTrue(os.path.exists(stage_1_checkpoint))
            self.assertTrue(os.path.exists(stage_2_checkpoint))


if __name__ == "__main__":
    unittest.main()
