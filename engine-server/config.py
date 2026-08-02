"""Configuration service for the engine server.

Loads the engine's YAML config, applies user overrides from user_config.yaml,
and persists updates. The engine server is the source of truth for any config
that affects the engine (balance, risk, pairs).
"""

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

ENGINE_ROOT = Path(__file__).resolve().parent.parent  # repo root
DEFAULT_CONFIG_PATH = ENGINE_ROOT / "configs" / "config.yaml"
USER_CONFIG_PATH = Path(__file__).resolve().parent / "user_config.yaml"

# Make the repo root importable regardless of the CWD the server is started from.
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

DEFAULT_BALANCE = 10000.0

# Top-level config sections the desktop may edit. Everything else (database,
# kafka, models internals) is owned by the engine and never persisted in the
# user overrides file.
USER_EDITABLE = ("account", "pairs", "risk", "simulation", "market")

_RISK_EDITABLE = ("sizing", "circuit_breakers", "limits", "monitoring")


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into a copy of base. Override wins."""
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _sanitize(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only sections the desktop is allowed to persist."""
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in USER_EDITABLE:
            continue
        if key == "risk" and isinstance(value, dict):
            out[key] = {k: v for k, v in value.items() if k in _RISK_EDITABLE}
        else:
            out[key] = deepcopy(value)
    return out


def _load_user_config() -> Dict[str, Any]:
    if USER_CONFIG_PATH.exists():
        with open(USER_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_user_config(data: Dict[str, Any]) -> None:
    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = USER_CONFIG_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    os.replace(tmp, USER_CONFIG_PATH)


def _load_engine_config() -> Dict[str, Any]:
    from configs.loader import load_config  # local import keeps boot light

    app_config = load_config(str(DEFAULT_CONFIG_PATH))
    if hasattr(app_config, "model_dump"):
        return app_config.model_dump()
    return app_config.dict()


def get_config() -> Dict[str, Any]:
    """Return the effective config: engine defaults + user overrides."""
    config = _load_engine_config()
    user = _load_user_config()

    # Inject account balance (surfaces as account.balance)
    account = user.get("account", {})
    config["account"] = account
    config.setdefault("account", {})
    if "balance" not in config["account"]:
        config["account"]["balance"] = DEFAULT_BALANCE

    merged = merge_dicts(config, user)
    return merged


def update_config(partial: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a sanitized partial into the persisted user config, then return
    the new effective config. Only desktop-editable sections are persisted."""
    partial = _sanitize(partial)
    user = _load_user_config()
    updated = merge_dicts(user, partial)
    _save_user_config(updated)
    return get_config()
