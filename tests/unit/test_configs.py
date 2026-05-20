import os
import unittest
from configs.loader import load_config, AppConfig


class TestConfigsLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.config_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path = os.path.join(self.config_dir, "configs", "config.example.yaml")

    def test_load_example_config(self) -> None:
        """
        Verify config.example.yaml can be successfully loaded and validated.
        """
        config = load_config(self.config_path)
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.environment, "development")
        self.assertEqual(config.database.redis.port, 6379)
        self.assertIn("EUR_USD", config.pairs)

    def test_env_var_override(self) -> None:
        """
        Verify that environment variables successfully override configuration fields.
        """
        os.environ["FOREX_ENVIRONMENT"] = "live"
        os.environ["FOREX_DATABASE_TIMESCALEDB_PORT"] = "9999"
        os.environ["FOREX_DATABASE_TIMESCALEDB_PASSWORD"] = "secret_env_pass"

        try:
            config = load_config(self.config_path)
            self.assertEqual(config.environment, "live")
            self.assertEqual(config.database.timescaledb.port, 9999)
            self.assertEqual(config.database.timescaledb.password, "secret_env_pass")
        finally:
            # Clean up env
            os.environ.pop("FOREX_ENVIRONMENT", None)
            os.environ.pop("FOREX_DATABASE_TIMESCALEDB_PORT", None)
            os.environ.pop("FOREX_DATABASE_TIMESCALEDB_PASSWORD", None)


    def test_modular_config_merging(self) -> None:
        """
        Verify that modular configurations in subdirectories (models, risk, brokers)
        are loaded and merged into the loaded AppConfig object.
        """
        config = load_config(self.config_path)
        
        # Verify model configs are merged
        self.assertEqual(config.models.temporal.seq_len, 60)
        self.assertEqual(config.models.temporal.d_model, 256)
        
        # Verify alternative feature configs are defaulted
        self.assertEqual(config.features.alternative.cot.cot_window, 30)
        
        # Verify risk sizing is merged
        self.assertEqual(config.risk.sizing.kelly_fraction, 0.25)
        self.assertEqual(config.risk.sizing.method, "kelly")
        
        # Verify risk monitoring is defaulted/merged
        self.assertEqual(config.risk.monitoring.max_drawdown, 0.10)
        self.assertEqual(config.risk.monitoring.alert_cooldown_seconds, 300.0)
        
        # Verify broker configs are merged
        self.assertEqual(config.execution.oanda.domain, "api-fxpractice.oanda.com")
        self.assertEqual(config.execution.lmax.port, 4001)
        self.assertEqual(config.execution.ib.client_id, 1)


if __name__ == "__main__":
    unittest.main()
