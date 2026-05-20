import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Optional, Dict, List
from models.base_model import BaseModel
from models.temporal.transformer import PositionalEncoding
from models.temporal.tcn import TCNResidualBlock


class TemporalFusionNet(nn.Module):
    """
    Inner PyTorch Neural Network executing Cross-Attention Fusion.
    Extracts global features via Transformer Encoder and local features via TCN,
    fusing them via multi-head Cross-Attention.
    """
    def __init__(
        self,
        d_feat: int,
        d_model: int = 64,
        num_channels: Optional[List[int]] = None,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        if num_channels is None:
            num_channels = [32, 64, 64]
            
        # 1. Transformer Backbone
        self.projection = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_backbone = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 2. TCN Backbone
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_chs = d_feat if i == 0 else num_channels[i - 1]
            out_chs = num_channels[i]
            layers.append(
                TCNResidualBlock(in_chs, out_chs, kernel_size, dilation_size, dropout)
            )
        self.tcn_backbone = nn.Sequential(*layers)
        
        # 3. Cross-Attention
        d_tcn = num_channels[-1]
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            kdim=d_tcn,
            vdim=d_tcn,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Output Projection
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input shape: [batch, seq_len, d_feat]
        
        # 1. Transformer pass
        x_trans = self.projection(x)  # [batch, seq_len, d_model]
        x_trans = self.pos_encoder(x_trans)  # [batch, seq_len, d_model]
        trans_out = self.transformer_backbone(x_trans, mask=mask)  # [batch, seq_len, d_model]
        
        # 2. TCN pass
        x_tcn_input = x.transpose(1, 2)  # [batch, d_feat, seq_len]
        tcn_out = self.tcn_backbone(x_tcn_input)  # [batch, d_tcn, seq_len]
        tcn_out = tcn_out.transpose(1, 2)  # [batch, seq_len, d_tcn]
        
        # 3. Cross-Attention
        # Query: Transformer (Global contexts)
        # Key/Value: TCN (Local causal features)
        fused, _ = self.cross_attention(
            query=trans_out,
            key=tcn_out,
            value=tcn_out,
            attn_mask=mask
        )  # [batch, seq_len, d_model]
        
        # Squeeze sequence to last element
        fused_last = fused[:, -1, :]  # [batch, d_model]
        return self.fc_out(fused_last).squeeze(-1)  # [batch]


class TemporalFusionModel(BaseModel):
    """
    SQL-compatible, BaseModel-compliant wrapper for PyTorch TemporalFusionNet.
    Fuses TCN & Transformer representations.
    Fits continuous forward returns (regression).
    """
    def __init__(self, name: str = "temporal_fusion", config: Any = None) -> None:
        super().__init__(name, config)
        
        # Parse hyperparams
        cfg = config.get("temporal_fusion", {}) if config else {}
        self.d_model = cfg.get("d_model", 64)
        self.num_channels = cfg.get("num_channels", [32, 64, 64])
        self.nhead = cfg.get("nhead", 4)
        self.num_layers = cfg.get("num_layers", 2)
        self.dim_feedforward = cfg.get("dim_feedforward", 128)
        self.kernel_size = cfg.get("kernel_size", 3)
        self.dropout = cfg.get("dropout", 0.1)
        self.lr = cfg.get("lr", 1e-3)
        self.epochs = cfg.get("epochs", 10)
        self.batch_size = cfg.get("batch_size", 64)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TemporalFusionNet] = None
        self.d_feat: Optional[int] = None

    def _init_network(self, d_feat: int) -> None:
        self.d_feat = d_feat
        self.model = TemporalFusionNet(
            d_feat=self.d_feat,
            d_model=self.d_model,
            num_channels=self.num_channels,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=self.device),
            diagonal=1
        )
        return mask

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "TemporalFusionModel":
        """
        Fits the TemporalFusionModel on sequence inputs X and target returns y.
        X shape: [n_samples, seq_len, d_feat]
        y shape: [n_samples]
        """
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        
        n_samples, seq_len, d_feat = X_tensor.shape
        
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
                
            epoch_loss /= n_samples
            logger.info(f"Epoch {epoch + 1}/{self.epochs} completed", loss=float(epoch_loss))
            
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        """
        Inference on sequence inputs X.
        X shape: [n_samples, seq_len, d_feat]
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        self.model.eval()
        
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
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
            "num_channels": self.num_channels,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "kernel_size": self.kernel_size,
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
        self.num_channels = state["num_channels"]
        self.nhead = state["nhead"]
        self.num_layers = state["num_layers"]
        self.dim_feedforward = state["dim_feedforward"]
        self.kernel_size = state["kernel_size"]
        self.dropout = state["dropout"]
        
        self._init_network(self.d_feat)
        self.model.load_state_dict(state["state_dict"])
