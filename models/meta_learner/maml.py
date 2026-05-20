import os
import ctypes
import sys
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import structlog

from models.base_model import BaseModel

logger = structlog.get_logger()

_SPEEDUPS_LIB = None
try:
    _base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _ext = ".dll" if sys.platform.startswith("win") else ".dylib" if sys.platform.startswith("darwin") else ".so"
    _lib_path = os.path.join(_base_dir, "models", "meta_learner", f"maml_speedups{_ext}")
    
    if os.path.exists(_lib_path):
        _SPEEDUPS_LIB = ctypes.CDLL(_lib_path)
        
        _SPEEDUPS_LIB.maml_inner_loop_update.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int
        ]
        _SPEEDUPS_LIB.maml_inner_loop_update.restype = None
        
        _SPEEDUPS_LIB.compute_linear_forward_and_gradient.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]
        _SPEEDUPS_LIB.compute_linear_forward_and_gradient.restype = None
        logger.info("Loaded MAML C++ speedups library successfully.")
    else:
        logger.warning(f"MAML C++ speedups library not found at {_lib_path}. Using pure Python fallback.")
except Exception as e:
    logger.warning(f"Failed to load MAML C++ speedups library: {e}. Using pure Python fallback.")


class MAMLNetwork(nn.Module):
    """
    Lightweight 3-layer MLP designed for fast few-shot adaptation in MAML.
    """
    def __init__(self, d_feat: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_feat, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x).squeeze(-1)


class MAMLModel(BaseModel):
    """
    Model-Agnostic Meta-Learning (MAML) wrapper for few-shot regime adaptation.
    """
    def __init__(self, name: str = "maml", config: Any = None) -> None:
        config = config or {}
        super().__init__(name=name, config=config)
        
        maml_cfg = config.get("maml", {})
        self.inner_lr = maml_cfg.get("inner_lr", 0.01)
        self.outer_lr = maml_cfg.get("outer_lr", 0.001)
        self.num_inner_steps = maml_cfg.get("num_inner_steps", 5)
        self.support_size = maml_cfg.get("support_size", 50)
        self.query_size = maml_cfg.get("query_size", 20)
        self.meta_batch_size = maml_cfg.get("meta_batch_size", 8)
        self.meta_epochs = maml_cfg.get("meta_epochs", 50)
        
        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        
        self.model: Optional[MAMLNetwork] = None
        self.d_feat: Optional[int] = None
        self.adapted_model: Optional[MAMLNetwork] = None
        
        logger.info(
            "MAMLModel initialized",
            inner_lr=self.inner_lr,
            outer_lr=self.outer_lr,
            num_inner_steps=self.num_inner_steps,
            device=str(self.device)
        )
        
    def _init_network(self, d_feat: int) -> None:
        self.d_feat = d_feat
        self.model = MAMLNetwork(d_feat=self.d_feat).to(self.device)
        
    def _construct_episodes(self, X: np.ndarray, y: np.ndarray) -> list:
        episodes = []
        n_samples = X.shape[0]
        episode_len = self.support_size + self.query_size
        stride = self.support_size // 2
        
        for i in range(0, n_samples - episode_len + 1, stride):
            support_end = i + self.support_size
            query_end = support_end + self.query_size
            
            s_X = torch.tensor(X[i:support_end], dtype=torch.float32).to(self.device)
            s_y = torch.tensor(y[i:support_end], dtype=torch.float32).to(self.device)
            q_X = torch.tensor(X[support_end:query_end], dtype=torch.float32).to(self.device)
            q_y = torch.tensor(y[support_end:query_end], dtype=torch.float32).to(self.device)
            
            episodes.append((s_X, s_y, q_X, q_y))
            
        return episodes

    def _clone_model(self, model: nn.Module) -> nn.Module:
        clone = MAMLNetwork(d_feat=self.d_feat).to(self.device)
        clone.load_state_dict(model.state_dict())
        for param in clone.parameters():
            param.requires_grad = True
        return clone
        
    def _inner_loop_adapt(self, model: nn.Module, support_X: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        clone = self._clone_model(model)
        optimizer = optim.SGD(clone.parameters(), lr=self.inner_lr)
        clone.train()
        
        for _ in range(self.num_inner_steps):
            optimizer.zero_grad()
            preds = clone(support_X)
            loss = nn.MSELoss()(preds, support_y)
            loss.backward()
            optimizer.step()
            
        return clone

    def _meta_train_step(self, episodes_batch: list, optimizer: optim.Optimizer) -> float:
        outer_loss = 0.0
        optimizer.zero_grad()
        
        for s_X, s_y, q_X, q_y in episodes_batch:
            clone = self._clone_model(self.model)
            inner_opt = optim.SGD(clone.parameters(), lr=self.inner_lr)
            clone.train()
            
            for _ in range(self.num_inner_steps):
                inner_opt.zero_grad()
                s_pred = clone(s_X)
                s_loss = nn.MSELoss()(s_pred, s_y)
                s_loss.backward()
                inner_opt.step()
                
            q_pred = clone(q_X)
            q_loss = nn.MSELoss()(q_pred, q_y)
            outer_loss += q_loss.item()
            
            for param, clone_param in zip(self.model.parameters(), clone.parameters()):
                if param.grad is None:
                    param.grad = torch.zeros_like(param.data)
                param.grad.data.add_(param.data - clone_param.data)
                
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.div_(len(episodes_batch))
                
        optimizer.step()
        return outer_loss / len(episodes_batch)

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "MAMLModel":
        if y is None:
            raise ValueError("y cannot be None for MAML meta-training.")
            
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        
        if self.model is None:
            self._init_network(X_arr.shape[-1])
            
        episodes = self._construct_episodes(X_arr, y_arr)
        if not episodes:
            logger.warning("Not enough data to construct MAML episodes.")
            return self
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.outer_lr)
        
        self.model.train()
        for epoch in range(self.meta_epochs):
            indices = np.random.permutation(len(episodes))
            epoch_loss = 0.0
            batches = 0
            
            for i in range(0, len(episodes), self.meta_batch_size):
                batch_idx = indices[i:i + self.meta_batch_size]
                batch = [episodes[j] for j in batch_idx]
                
                loss = self._meta_train_step(batch, optimizer)
                epoch_loss += loss
                batches += 1
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "MAML Meta-training epoch",
                    epoch=epoch + 1,
                    avg_query_loss=epoch_loss / batches
                )
                
        return self

    def adapt(self, X_support: Any, y_support: Any) -> None:
        X_arr = np.asarray(X_support, dtype=np.float32)
        y_arr = np.asarray(y_support, dtype=np.float32)
        
        s_X = torch.tensor(X_arr).to(self.device)
        s_y = torch.tensor(y_arr).to(self.device)
        
        if self.model is None:
            self._init_network(s_X.shape[-1])
            
        self.adapted_model = self._inner_loop_adapt(self.model, s_X, s_y)
        
    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X_arr).to(self.device)
        
        eval_model = getattr(self, "adapted_model", None)
        if eval_model is None:
            eval_model = self.model
        if eval_model is None:
            return np.zeros(X_arr.shape[0])
            
        eval_model.eval()
        with torch.no_grad():
            preds = eval_model(X_tensor)
            
        return preds.cpu().numpy()

    def save(self, path: str, **kwargs: Any) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "name": self.name,
            "config": self.config,
            "d_feat": self.d_feat,
            "inner_lr": self.inner_lr,
            "outer_lr": self.outer_lr,
            "num_inner_steps": self.num_inner_steps,
            "model_state": self.model.state_dict() if self.model else None
        }
        torch.save(state, path)
        logger.info("MAMLModel saved successfully", destination=path)

    def load(self, path: str, **kwargs: Any) -> None:
        state = torch.load(path, map_location=self.device)
        self.name = state["name"]
        self.config = state["config"]
        self.d_feat = state["d_feat"]
        self.inner_lr = state["inner_lr"]
        self.outer_lr = state["outer_lr"]
        self.num_inner_steps = state["num_inner_steps"]
        
        if self.d_feat is not None:
            self._init_network(self.d_feat)
            if state["model_state"] is not None:
                self.model.load_state_dict(state["model_state"])
        logger.info("MAMLModel loaded successfully", source=path)
