import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import structlog
from typing import Any, Optional, Dict

logger = structlog.get_logger()
from models.base_model import BaseModel


class LSTMRegimeNet(nn.Module):
    """
    Inner PyTorch Neural Network executing LSTM sequence classification.
    """
    def __init__(
        self,
        d_feat: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, d_feat]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        last_out = lstm_out[:, -1, :]  # [batch, hidden_dim]
        out = self.fc_out(self.dropout(last_out))  # [batch, num_classes]
        return out


class LSTMRegimeClassifier(BaseModel):
    """
    SQL-compatible, BaseModel-compliant wrapper for PyTorch LSTM Regime Classifier.
    Predicts regime probabilities based on historical rolling sequence windows.
    Trained on pseudo-labels (typically from unsupervised HMM).
    """
    def __init__(self, name: str = "lstm_regime", config: Any = None) -> None:
        super().__init__(name, config)
        
        cfg = config.get("lstm_regime", {}) if config else {}
        self.hidden_dim = cfg.get("hidden_dim", 64)
        self.num_layers = cfg.get("num_layers", 2)
        self.num_classes = cfg.get("num_classes", 4)
        self.dropout = cfg.get("dropout", 0.1)
        self.lr = cfg.get("lr", 1e-3)
        self.epochs = cfg.get("epochs", 10)
        self.batch_size = cfg.get("batch_size", 64)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMRegimeNet] = None
        self.d_feat: Optional[int] = None

    def _init_network(self, d_feat: int) -> None:
        self.d_feat = d_feat
        self.model = LSTMRegimeNet(
            d_feat=self.d_feat,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "LSTMRegimeClassifier":
        """
        Fits the LSTM regime classifier on sequence inputs X and targets y (class indices).
        X shape: [n_samples, seq_len, d_feat]
        y shape: [n_samples] (values in range [0, num_classes-1])
        """
        if y is None:
            raise ValueError("Supervised training target (y) must be provided.")
            
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.long)
        
        n_samples, seq_len, d_feat = X_tensor.shape
        
        if self.model is None or self.d_feat != d_feat:
            self._init_network(d_feat)
            
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_x)
                
            epoch_loss /= n_samples
            logger.info(f"Regime LSTM - Epoch {epoch + 1}/{self.epochs} completed", loss=float(epoch_loss))
            
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        """
        Predicts class labels or probabilities for inputs X.
        By default, returns class index predictions (arg max).
        If return_proba=True, returns probability distributions.
        X shape: [n_samples, seq_len, d_feat] or [seq_len, d_feat]
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        self.model.eval()
        
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(0)
            
        return_proba = kwargs.get("return_proba", False)
        
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            logits = self.model(X_tensor)
            if return_proba:
                probs = F.softmax(logits, dim=-1)
                return probs.cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=-1)
                return preds.cpu().numpy()

    def save(self, path: str, **kwargs: Any) -> None:
        """Saves PyTorch weights and configurations to disk."""
        if self.model is None:
            raise ValueError("No model state to save.")
            
        state = {
            "d_feat": self.d_feat,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "state_dict": self.model.state_dict()
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

    def load(self, path: str, **kwargs: Any) -> None:
        """Loads PyTorch weights and configurations from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"State file not found at: {path}")
            
        state = torch.load(path, map_location=self.device)
        self.d_feat = state["d_feat"]
        self.hidden_dim = state["hidden_dim"]
        self.num_layers = state["num_layers"]
        self.num_classes = state["num_classes"]
        self.dropout = state["dropout"]
        
        self._init_network(self.d_feat)
        self.model.load_state_dict(state["state_dict"])
