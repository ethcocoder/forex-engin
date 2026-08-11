import numpy as np
import pandas as pd
import structlog
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

logger = structlog.get_logger()

class ModelTrainingHarness:
    """
    Reproducible experiment tracking and model training harness for candidate models.
    Trains on leakage-safe walk-forward splits and logs out-of-sample metrics.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_baseline_classifier(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains a baseline Random Forest classifier on walk-forward folds.
        """
        feature_cols = [c for c in train_df.columns if c not in ["timestamp", "bid", "ask", "mid", "label"]]
        
        # Create a directional label if not present (1 if next mid return > 0 else 0)
        if "label" not in train_df.columns:
            train_df = train_df.copy()
            train_df["label"] = (train_df["return_1"].shift(-1) > 0).astype(int)
            train_df = train_df.dropna()

        if "label" not in test_df.columns:
            test_df = test_df.copy()
            test_df["label"] = (test_df["return_1"].shift(-1) > 0).astype(int)
            test_df = test_df.dropna()

        if train_df.empty or test_df.empty:
            return {"status": "SKIPPED", "reason": "Insufficient data after labeling"}

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_test = test_df[feature_cols]
        y_test = test_df["label"]

        model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)

        metrics = {
            "status": "SUCCESS",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": float(acc),
            "precision": float(prec)
        }
        logger.info("Model training harness evaluation completed", metrics=metrics)
        return metrics

if __name__ == "__main__":
    from models.feature_pipeline import LeakageSafeFeaturePipeline
    
    sample_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-01", periods=600, freq="s", tz="UTC"),
        "bid": np.linspace(1.1000, 1.1600, 600) + np.random.normal(0, 0.0001, 600),
        "ask": np.linspace(1.1002, 1.1602, 600) + np.random.normal(0, 0.0001, 600),
        "mid": np.linspace(1.1001, 1.1601, 600) + np.random.normal(0, 0.0001, 600)
    })
    
    pipe = LeakageSafeFeaturePipeline()
    feat_df = pipe.compute_features(sample_df)
    splits = pipe.purged_walk_forward_split(feat_df, n_splits=2, purge_window=5)
    
    if splits:
        train_fold, test_fold = splits[0]
        harness = ModelTrainingHarness()
        res = harness.train_baseline_classifier(train_fold, test_fold)
        print("Training Harness Result:", res)
