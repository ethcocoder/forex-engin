import os
from typing import List, Literal, Optional
import yaml
from pydantic import BaseModel, Field, ValidationError, ConfigDict


class TimescaleConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    username: str = Field(default="postgres")
    password: str = Field(default="secure_password")
    database: str = Field(default="forex_db")
    pool_min: int = Field(default=5)
    pool_max: int = Field(default=20)


class RedisConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    password: str = Field(default="")
    db: int = Field(default=0)
    max_tick_lookback: int = Field(default=1000)


class DatabaseConfig(BaseModel):
    timescaledb: TimescaleConfig = Field(default_factory=TimescaleConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)


class KafkaConfig(BaseModel):
    bootstrap_servers: List[str] = Field(default_factory=lambda: ["localhost:9092"])
    ticks_topic: str = Field(default="forex.ticks", alias="topics.ticks")
    features_topic: str = Field(default="forex.features", alias="topics.features")
    signals_topic: str = Field(default="forex.signals", alias="topics.signals")
    orders_topic: str = Field(default="forex.orders", alias="topics.orders")
    fills_topic: str = Field(default="forex.fills", alias="topics.fills")
    consumer_group: str = Field(default="forex-neural-pipeline")

    model_config = ConfigDict(populate_by_name=True)


class ResamplingConfig(BaseModel):
    base_timeframe: str = Field(default="1m")
    aggregates: List[str] = Field(default_factory=lambda: ["5m", "1h", "4h", "1d"])


class VPINConfig(BaseModel):
    volume_bucket_size: int = Field(default=50)


class KyleLambdaConfig(BaseModel):
    window_size: int = Field(default=100)


class AmihudConfig(BaseModel):
    window_size: int = Field(default=60)


class MicrostructureConfig(BaseModel):
    vpin: VPINConfig = Field(default_factory=VPINConfig)
    kyle_lambda: KyleLambdaConfig = Field(default_factory=KyleLambdaConfig)
    amihud: AmihudConfig = Field(default_factory=AmihudConfig)


class WaveletConfig(BaseModel):
    wavelet_name: str = Field(default="db4")
    level: int = Field(default=5)


class KalmanConfig(BaseModel):
    process_noise: float = Field(default=1e-4)
    observation_noise: float = Field(default=1e-2)


class COTConfig(BaseModel):
    cot_window: int = Field(default=30)


class SentimentConfig(BaseModel):
    ema_span: int = Field(default=12)
    volume_window: int = Field(default=20)


class MacroConfig(BaseModel):
    momentum_window: int = Field(default=20)
    decay_factor: float = Field(default=0.95)


class OptionsConfig(BaseModel):
    rolling_window: int = Field(default=20)


class AlternativeConfig(BaseModel):
    cot: COTConfig = Field(default_factory=COTConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    macro: MacroConfig = Field(default_factory=MacroConfig)
    options: OptionsConfig = Field(default_factory=OptionsConfig)


class FeaturesConfig(BaseModel):
    resampling: ResamplingConfig = Field(default_factory=ResamplingConfig)
    microstructure: MicrostructureConfig = Field(default_factory=MicrostructureConfig)
    wavelet: WaveletConfig = Field(default_factory=WaveletConfig)
    kalman: KalmanConfig = Field(default_factory=KalmanConfig)
    alternative: AlternativeConfig = Field(default_factory=AlternativeConfig)


class TemporalModelConfig(BaseModel):
    seq_len: int = Field(default=60)
    d_model: int = Field(default=256)
    nhead: int = Field(default=8)
    num_layers: int = Field(default=4)
    dim_feedforward: int = Field(default=512)
    dropout: float = Field(default=0.1)


class RegimeModelConfig(BaseModel):
    n_states: int = Field(default=4)
    hmm_features: List[str] = Field(
        default_factory=lambda: [
            "volatility_cc",
            "mean_reversion_hurst",
            "trend_strength",
            "vpin",
        ]
    )


class RLAgentConfig(BaseModel):
    algorithm: Literal["PPO", "SAC"] = Field(default="PPO")
    learning_rate: float = Field(default=3e-4)
    n_steps: int = Field(default=2048)
    batch_size: int = Field(default=64)
    ent_coef: float = Field(default=0.01)
    gamma: float = Field(default=0.99)
    gae_lambda: float = Field(default=0.95)


class MetaLearnerConfig(BaseModel):
    adaptation_steps: int = Field(default=5)
    support_set_size: int = Field(default=50)
    inner_lr: float = Field(default=1e-2)
    outer_lr: float = Field(default=1e-4)


class EnsembleConfig(BaseModel):
    aggregation_method: Literal["stacking", "bma"] = Field(default="stacking")
    mc_dropout_samples: int = Field(default=50)
    stacking_meta_model: str = Field(default="lightgbm")


class ModelsConfig(BaseModel):
    temporal: TemporalModelConfig = Field(default_factory=TemporalModelConfig)
    regime: RegimeModelConfig = Field(default_factory=RegimeModelConfig)
    rl_agent: RLAgentConfig = Field(default_factory=RLAgentConfig)
    meta_learner: MetaLearnerConfig = Field(default_factory=MetaLearnerConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)


class SizingConfig(BaseModel):
    method: Literal["kelly", "fixed", "volatility"] = Field(default="kelly")
    kelly_fraction: float = Field(default=0.25)
    max_account_risk_pct: float = Field(default=0.02)


class CircuitBreakersConfig(BaseModel):
    daily_drawdown_limit: float = Field(default=0.03)
    weekly_drawdown_limit: float = Field(default=0.06)
    monthly_drawdown_limit: float = Field(default=0.10)


class LimitsConfig(BaseModel):
    cvar_confidence: float = Field(default=0.95)
    max_open_positions: int = Field(default=5)
    max_correlation_exposure: float = Field(default=0.70)


class MonitoringConfig(BaseModel):
    max_drawdown: float = Field(default=0.10)
    max_leverage: float = Field(default=10.0)
    max_concentration: float = Field(default=0.30)
    var_confidence: float = Field(default=0.95)
    alert_cooldown_seconds: float = Field(default=300.0)


class RiskConfig(BaseModel):
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    circuit_breakers: CircuitBreakersConfig = Field(
        default_factory=CircuitBreakersConfig
    )
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


class SlippageConfig(BaseModel):
    base_spread_pct: float = Field(default=0.0001)
    market_impact_coefficient: float = Field(default=0.05)


class SmartRoutingConfig(BaseModel):
    routing_enabled: bool = Field(default=True)
    twap_slice_interval_sec: int = Field(default=60)
    vwap_volume_participation_rate: float = Field(default=0.10)


class OandaConfig(BaseModel):
    access_token: str = Field(default="")
    account_id: str = Field(default="")
    domain: str = Field(default="api-fxpractice.oanda.com")


class LMAXConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=4001)
    sender_comp_id: str = Field(default="CLIENT_SENDER")
    target_comp_id: str = Field(default="LMAX_TARGET")
    username: str = Field(default="lmax_user")
    password: str = Field(default="lmax_pass")
    initial_capital: float = Field(default=100000.0)


class IBConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=7497)
    client_id: int = Field(default=1)
    initial_capital: float = Field(default=100000.0)


class ExecutionConfig(BaseModel):
    broker: Literal["paper", "oanda", "lmax", "ib"] = Field(default="paper")
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    smart_routing: SmartRoutingConfig = Field(default_factory=SmartRoutingConfig)
    oanda: OandaConfig = Field(default_factory=OandaConfig)
    lmax: LMAXConfig = Field(default_factory=LMAXConfig)
    ib: IBConfig = Field(default_factory=IBConfig)


class AppConfig(BaseModel):
    environment: Literal["development", "paper", "live"] = Field(default="development")
    pairs: List[str] = Field(default_factory=lambda: ["EUR_USD", "GBP_USD"])
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


def load_config(config_path: str) -> AppConfig:
    """
    Loads, merges, and validates configuration from a YAML file and overrides using Environment Variables.
    Also merges modular configuration files from models/, risk/, and brokers/ subdirectories if present.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    config_dir = os.path.dirname(config_path)

    # 1. Merge modular model configs: configs/models/*.yaml -> models.*
    models_dir = os.path.join(config_dir, "models")
    if os.path.isdir(models_dir):
        if "models" not in data:
            data["models"] = {}
        
        # Mapping from file name to models config sub-key
        model_file_mappings = {
            "temporal_model.yaml": "temporal",
            "regime_model.yaml": "regime",
            "rl_agent.yaml": "rl_agent",
            "meta_learner.yaml": "meta_learner",
            "ensemble.yaml": "ensemble"
        }
        for filename, subkey in model_file_mappings.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r") as f:
                    model_data = yaml.safe_load(f) or {}
                if model_data:
                    if subkey not in data["models"]:
                        data["models"][subkey] = {}
                    data["models"][subkey].update(model_data)

    # 2. Merge modular risk configs: configs/risk/*.yaml -> risk.*
    risk_dir = os.path.join(config_dir, "risk")
    if os.path.isdir(risk_dir):
        if "risk" not in data:
            data["risk"] = {}
        
        # position_sizing.yaml -> risk.sizing
        sizing_path = os.path.join(risk_dir, "position_sizing.yaml")
        if os.path.isfile(sizing_path):
            with open(sizing_path, "r") as f:
                sizing_data = yaml.safe_load(f) or {}
            if sizing_data:
                if "sizing" not in data["risk"]:
                    data["risk"]["sizing"] = {}
                data["risk"]["sizing"].update(sizing_data)

        # risk_limits.yaml -> risk.limits and risk.circuit_breakers
        limits_path = os.path.join(risk_dir, "risk_limits.yaml")
        if os.path.isfile(limits_path):
            with open(limits_path, "r") as f:
                limits_data = yaml.safe_load(f) or {}
            if limits_data:
                cb_keys = {"daily_drawdown_limit", "weekly_drawdown_limit", "monthly_drawdown_limit"}
                mon_keys = {"max_drawdown", "max_leverage", "max_concentration", "var_confidence", "alert_cooldown_seconds"}
                
                has_nested = any(k in limits_data for k in ["limits", "circuit_breakers", "monitoring"])
                if has_nested:
                    for k in ["limits", "circuit_breakers", "monitoring"]:
                        if k in limits_data and isinstance(limits_data[k], dict):
                            if k not in data["risk"]:
                                data["risk"][k] = {}
                            data["risk"][k].update(limits_data[k])
                else:
                    for k, v in limits_data.items():
                        if k in cb_keys:
                            if "circuit_breakers" not in data["risk"]:
                                data["risk"]["circuit_breakers"] = {}
                            data["risk"]["circuit_breakers"][k] = v
                        elif k in mon_keys:
                            if "monitoring" not in data["risk"]:
                                data["risk"]["monitoring"] = {}
                            data["risk"]["monitoring"][k] = v
                        else:
                            if "limits" not in data["risk"]:
                                data["risk"]["limits"] = {}
                            data["risk"]["limits"][k] = v

    # 3. Merge modular broker configs: configs/brokers/*.yaml -> execution.*
    brokers_dir = os.path.join(config_dir, "brokers")
    if os.path.isdir(brokers_dir):
        if "execution" not in data:
            data["execution"] = {}
        
        broker_file_mappings = {
            "oanda.yaml": "oanda",
            "lmax.yaml": "lmax",
            "interactive_brokers.yaml": "ib"
        }
        for filename, subkey in broker_file_mappings.items():
            filepath = os.path.join(brokers_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r") as f:
                    broker_data = yaml.safe_load(f) or {}
                if broker_data:
                    if subkey not in data["execution"]:
                        data["execution"][subkey] = {}
                    data["execution"][subkey].update(broker_data)

    # Extract nested fields and load flattened representations for Kafka topics
    if "kafka" in data and "topics" in data["kafka"]:
        topics = data["kafka"]["topics"]
        for topic_key, topic_val in topics.items():
            data["kafka"][f"topics.{topic_key}"] = topic_val

    # Apply environment overrides before validation
    _apply_env_overrides(data)

    try:
        config = AppConfig(**data)
        return config
    except ValidationError as e:
        print(f"CRITICAL: Configuration validation failed for: {config_path}")
        raise e


def _apply_env_overrides(data: dict, prefix: str = "FOREX") -> None:
    """
    Traverses environment variables starting with a prefix and maps them onto the nested config dictionary.
    Format: FOREX_DATABASE_TIMESCALEDB_PASSWORD maps to data['database']['timescaledb']['password']
    """
    for key, val in os.environ.items():
        if key.startswith(f"{prefix}_"):
            parts = key[len(prefix) + 1 :].lower().split("_")
            curr = data
            for i, part in enumerate(parts[:-1]):
                if part not in curr:
                    curr[part] = {}
                elif not isinstance(curr[part], dict):
                    # Edge-case: if intermediate path was typed differently, reset
                    curr[part] = {}
                curr = curr[part]

            # Set value and typecast appropriately
            last_part = parts[-1]
            if val.lower() in ("true", "yes", "on"):
                curr[last_part] = True
            elif val.lower() in ("false", "no", "off"):
                curr[last_part] = False
            elif val.isdigit():
                curr[last_part] = int(val)
            else:
                try:
                    curr[last_part] = float(val)
                except ValueError:
                    curr[last_part] = val
