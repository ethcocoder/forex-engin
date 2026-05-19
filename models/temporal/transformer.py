import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Optional, Dict
from models.base_model import BaseModel


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding to inject sequence order context.
    """
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderNet(nn.Module):
    """
    Inner PyTorch Neural Network executing causal self-attention.
    """
    def __init__(
        self,
        d_feat: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input shape: [batch, seq_len, d_feat]
        x = self.projection(x)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)  # [batch, seq_len, d_model]
        
        # Apply transformer encoder with causal masking
        out = self.transformer_encoder(x, mask=mask)  # [batch, seq_len, d_model]
        
        # Extract features of the last sequence element
        out_last = out[:, -1, :]  # [batch, d_model]
        return self.fc_out(out_last).squeeze(-1)  # [batch]


class TransformerEncoderModel(BaseModel):
    """
    SQL-compatible, BaseModel-compliant wrapper for PyTorch Transformer Encoder.
    Fits continuous forward returns (regression).
    """
    def __init__(self, name: str = "transformer", config: Any = None) -> None:
        super().__init__(name, config)
        
        # Parse hyperparams
        cfg = config.get("transformer", {}) if config else {}
        self.d_model = cfg.get("d_model", 64)
        self.nhead = cfg.get("nhead", 4)
        self.num_layers = cfg.get("num_layers", 2)
        self.dim_feedforward = cfg.get("dim_feedforward", 128)
        self.dropout = cfg.get("dropout", 0.1)
        self.lr = cfg.get("lr", 1e-3)
        self.epochs = cfg.get("epochs", 10)
        self.batch_size = cfg.get("batch_size", 64)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TransformerEncoderNet] = None
        self.d_feat: Optional[int] = None

    def _init_network(self, d_feat: int) -> None:
        self.d_feat = d_feat
        self.model = TransformerEncoderNet(
            d_feat=self.d_feat,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=self.device),
            diagonal=1
        )
        return mask

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "TransformerEncoderModel":
        """
        Fits the transformer model on rolling sequence inputs X and target returns y.
        X shape: [n_samples, seq_len, d_feat]
        y shape: [n_samples]
        """
        # Convert inputs to torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        
        n_samples, seq_len, d_feat = X_tensor.shape
        
        # Initialize network if not already done
        if self.model is None or self.d_feat != d_feat:
            self._init_network(d_feat)
            
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        causal_mask = self._generate_causal_mask(seq_len)
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(batch_x, mask=causal_mask)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_x)
            
            # Print periodic metrics if validation logs are needed
            # logger.info("Epoch completed", epoch=epoch, mean_loss=epoch_loss/n_samples)
            
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        """
        Inference on sequence inputs X.
        X shape: [n_samples, seq_len, d_feat] or single sample [seq_len, d_feat]
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        self.model.eval()
        
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        
        # Add batch dimension if single sequence passed
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(0)
            
        n_samples, seq_len, d_feat = X_tensor.shape
        causal_mask = self._generate_causal_mask(seq_len)
        
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            preds = self.model(X_tensor, mask=causal_mask)
            return preds.cpu().numpy()

    def save(self, path: str, **kwargs: Any) -> None:
        """Saves PyTorch weights and configurations to disk."""
        if self.model is None:
            raise ValueError("No model state to save.")
            
        state = {
            "d_feat": self.d_feat,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
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
        self.d_model = state["d_model"]
        self.nhead = state["nhead"]
        self.num_layers = state["num_layers"]
        self.dim_feedforward = state["dim_feedforward"]
        self.dropout = state["dropout"]
        
        self._init_network(self.d_feat)
        self.model.load_state_dict(state["state_dict"])
