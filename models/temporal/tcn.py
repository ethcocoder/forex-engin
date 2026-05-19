import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Optional, Dict, List
from models.base_model import BaseModel


class CausalDilatedConv1D(nn.Module):
    """
    1D Causal Dilated Convolution.
    Pads input on the left to ensure output t only depends on inputs up to t.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,  # Manual padding done in forward
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, channels, seq_len]
        # Pad left side of the sequence length dimension
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class TCNResidualBlock(nn.Module):
    """
    Residual Block containing two Dilated Causal Convolution layers.
    Includes Batch Normalization, ReLU activations, Dropout, and a downsample path.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.conv1 = CausalDilatedConv1D(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = CausalDilatedConv1D(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        
        self.residual = nn.Sequential(
            self.conv1, self.norm1, self.relu1, self.drop1,
            self.conv2, self.norm2, self.relu2, self.drop2
        )
        
        # Match channel dimensions for residual sum if necessary
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.residual(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNNet(nn.Module):
    """
    Inner PyTorch TCN Network.
    Stack of residual blocks with exponentially increasing dilation.
    """
    def __init__(
        self,
        d_feat: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_chs = d_feat if i == 0 else num_channels[i - 1]
            out_chs = num_channels[i]
            layers.append(
                TCNResidualBlock(in_chs, out_chs, kernel_size, dilation_size, dropout)
            )
        self.tcn = nn.Sequential(*layers)
        self.fc_out = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, seq_len, d_feat]
        # Transpose to match Conv1d expected shape: [batch, d_feat, seq_len]
        x = x.transpose(1, 2)
        
        out = self.tcn(x)  # [batch, num_channels[-1], seq_len]
        
        # Take the output of the last sequence element
        out_last = out[:, :, -1]  # [batch, num_channels[-1]]
        return self.fc_out(out_last).squeeze(-1)  # [batch]


class TCNModel(BaseModel):
    """
    SQL-compatible, BaseModel-compliant wrapper for PyTorch TCN Model.
    Fits continuous forward returns (regression).
    """
    def __init__(self, name: str = "tcn", config: Any = None) -> None:
        super().__init__(name, config)
        
        # Parse hyperparams
        cfg = config.get("tcn", {}) if config else {}
        self.num_channels = cfg.get("num_channels", [32, 64, 64])
        self.kernel_size = cfg.get("kernel_size", 3)
        self.dropout = cfg.get("dropout", 0.2)
        self.lr = cfg.get("lr", 1e-3)
        self.epochs = cfg.get("epochs", 10)
        self.batch_size = cfg.get("batch_size", 64)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TCNNet] = None
        self.d_feat: Optional[int] = None

    def _init_network(self, d_feat: int) -> None:
        self.d_feat = d_feat
        self.model = TCNNet(
            d_feat=self.d_feat,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "TCNModel":
        """
        Fits the TCN model on rolling sequence inputs X and target returns y.
        X shape: [n_samples, seq_len, d_feat]
        y shape: [n_samples]
        """
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
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_x)
                
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
            
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            preds = self.model(X_tensor)
            return preds.cpu().numpy()

    def save(self, path: str, **kwargs: Any) -> None:
        """Saves PyTorch weights and configurations to disk."""
        if self.model is None:
            raise ValueError("No model state to save.")
            
        state = {
            "d_feat": self.d_feat,
            "num_channels": self.num_channels,
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
        self.num_channels = state["num_channels"]
        self.kernel_size = state["kernel_size"]
        self.dropout = state["dropout"]
        
        self._init_network(self.d_feat)
        self.model.load_state_dict(state["state_dict"])
