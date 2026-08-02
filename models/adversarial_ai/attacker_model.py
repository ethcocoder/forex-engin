import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim

from models.base_model import BaseModel


def _resolve_model_paths(base_path: str) -> Dict[str, Path]:
    base = Path(base_path)
    if base.suffix == ".pt":
        base = base.with_suffix("")

    return {
        "model_file": base.with_suffix(".pt"),
        "meta_file": base.with_suffix(".pkl"),
    }


class AttackerNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AttackerModel(BaseModel):
    """Adversarial attacker model that learns market vulnerability patterns."""

    def __init__(
        self,
        name: str = "adversarial_attacker",
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ) -> None:
        super().__init__(name, config or {})
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dims = self.config.get("hidden_dims", [256, 128, 64])
        self.feature_names = self.config.get("feature_names", [])
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.input_dim: Optional[int] = None
        self.model: Optional[AttackerNetwork] = None
        self.model_base = model_path or self.config.get("model_base", "saved_models/adversarial_attacker")
        self.model_file = None
        self.meta_file = None
        self._set_model_paths(self.model_base)

        if model_path and Path(model_path).exists():
            self.load(model_path)
        elif Path(self.model_file).exists() and Path(self.meta_file).exists():
            self.load(self.model_base)

    def _set_model_paths(self, base_path: str) -> None:
        paths = _resolve_model_paths(base_path)
        self.model_file = paths["model_file"]
        self.meta_file = paths["meta_file"]

    def _build_model(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.model = AttackerNetwork(input_dim=input_dim, hidden_dims=self.hidden_dims).to(self.device)

    def fit(
        self,
        X: Any,
        y: Any,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> "AttackerModel":
        X_arr = np.nan_to_num(np.asarray(X, dtype=np.float32), 0.0)
        y_arr = np.nan_to_num(np.asarray(y, dtype=np.float32), 0.0)

        if X_arr.ndim != 2:
            raise ValueError("AttackerModel expects 2D feature arrays of shape [n_samples, n_features].")

        if self.input_dim is None or self.input_dim != X_arr.shape[1]:
            self._build_model(X_arr.shape[1])

        self.scaler_mean = X_arr.mean(axis=0)
        self.scaler_std = X_arr.std(axis=0)
        self.scaler_std[self.scaler_std == 0.0] = 1.0

        X_scaled = (X_arr - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, 0.0)

        pos_weight = 1.0
        if len(y_arr) > 0:
            positives = float(np.count_nonzero(y_arr == 1.0))
            negatives = float(np.count_nonzero(y_arr == 0.0))
            if positives > 0:
                pos_weight = max(1.0, negatives / positives)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_scaled),
            torch.from_numpy(y_arr),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        assert self.model is not None
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=self.device))

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses) if epoch_losses else 0.0)
            logger = structlog.get_logger(__name__)
            logger.info("Adversarial attacker training epoch complete", epoch=epoch, loss=avg_loss)

        return self

    def predict(self, X: Any, return_proba: bool = True, **kwargs: Any) -> Any:
        if self.model is None or self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("Adversarial attacker model is not trained or loaded.")

        X_arr = np.nan_to_num(np.asarray(X, dtype=np.float32), 0.0)
        if X_arr.ndim == 1:
            X_arr = np.expand_dims(X_arr, axis=0)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.input_dim:
            raise ValueError(f"Input feature matrix must have shape [n_samples, {self.input_dim}].")

        X_scaled = (X_arr - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, 0.0)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_scaled).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()

        if return_proba:
            return probs
        return (probs > 0.5).astype(np.int32)

    def save(self, path: str, **kwargs: Any) -> None:
        self._set_model_paths(path)
        if self.model is None:
            raise RuntimeError("No trained adversarial model available to save.")

        os.makedirs(self.model_file.parent, exist_ok=True)
        torch.save(self.model.state_dict(), str(self.model_file))

        meta = {
            "name": self.name,
            "config": self.config,
            "device": self.device,
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
        }
        with open(self.meta_file, "wb") as f:
            pickle.dump(meta, f)

    def load(self, path: str, **kwargs: Any) -> None:
        self._set_model_paths(path)
        if not self.model_file.exists() or not self.meta_file.exists():
            raise FileNotFoundError(f"Adversarial model files not found at {self.model_file} and {self.meta_file}")

        with open(self.meta_file, "rb") as f:
            meta = pickle.load(f)

        self.name = meta.get("name", self.name)
        self.config = meta.get("config", self.config)
        self.device = meta.get("device", self.device)
        self.hidden_dims = meta.get("hidden_dims", self.hidden_dims)
        self.feature_names = meta.get("feature_names", self.feature_names)
        self.scaler_mean = meta.get("scaler_mean")
        self.scaler_std = meta.get("scaler_std")
        self.input_dim = int(meta.get("input_dim", self.input_dim))

        self._build_model(self.input_dim)
        assert self.model is not None
        self.model.load_state_dict(torch.load(str(self.model_file), map_location=self.device))
        self.model.to(self.device)

    def generate_adversarial_scenario(self, current_strategy: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None or self.scaler_mean is None or self.scaler_std is None:
            return {
                "status": "untrained",
                "message": "Adversarial attacker model is not trained. Train it with scripts/train_adversarial.py.",
            }

        candidate_count = self.config.get("candidate_count", 32)
        candidate_noise = self.config.get("candidate_noise", 0.8)

        with torch.no_grad():
            z = torch.randn(candidate_count, self.input_dim, device=self.device)
            raw_candidates = z * torch.from_numpy(self.scaler_std).to(self.device) * candidate_noise
            raw_candidates += torch.from_numpy(self.scaler_mean).to(self.device)
            logits = self.model(raw_candidates)
            probs = torch.sigmoid(logits).cpu().numpy()

        best_idx = int(np.argmax(probs))
        top_score = float(probs[best_idx])
        top_candidate = raw_candidates[best_idx].cpu().numpy()

        shock_indices = np.argsort(np.abs(top_candidate - self.scaler_mean))[-5:][::-1]
        top_shocks = {}
        for idx in shock_indices:
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            top_shocks[feature_name] = float(top_candidate[idx] - self.scaler_mean[idx])

        return {
            "strategy": current_strategy,
            "vulnerability_score": top_score,
            "expected_drawdown_pct": float(top_score * 0.05),
            "critical_feature_shocks": top_shocks,
            "status": "adversarial_scenario_generated",
        }
