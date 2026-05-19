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


if __name__ == "__main__":
    unittest.main()
