from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from models.feature_pipeline import LeakageSafeFeaturePipeline

logger = structlog.get_logger()


class ModelTrainingHarness:
    """Reproducible, out-of-sample evaluation harness for research candidates.

    This harness does not persist a production model. It evaluates an expanding
    walk-forward sequence and reports metrics based on executable bid/ask returns
    already attached by ``LeakageSafeFeaturePipeline``. A result is research
    evidence only; it does not qualify a system for broker-demo or live trading.
    """

    NON_FEATURE_COLUMNS = {
        "timestamp",
        "label",
        "label_end_timestamp",
        "label_horizon_bars",
        "long_return_h",
        "short_return_h",
        "instrument",
        "open",
        "high",
        "low",
        "close",
        "bid",
        "ask",
        "mid",
        "tick_count",
        "bid_volume",
        "ask_volume",
        "spread",
    }

    def __init__(self, random_state: int = 42, n_estimators: int = 200) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators

    def _feature_columns(self, df: pd.DataFrame) -> List[str]:
        feature_cols = [column for column in df.columns if column not in self.NON_FEATURE_COLUMNS]
        if not feature_cols:
            raise ValueError("No derived features available for training")
        non_numeric = [column for column in feature_cols if not pd.api.types.is_numeric_dtype(df[column])]
        if non_numeric:
            raise ValueError(f"Feature columns must be numeric: {non_numeric}")
        return feature_cols

    def train_baseline_classifier(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit a three-action baseline only on a single chronological training fold."""
        required = {
            "label",
            "long_return_h",
            "short_return_h",
            "label_end_timestamp",
            "timestamp",
        }
        missing = required - set(train_df.columns) - set(test_df.columns)
        if missing:
            raise ValueError(f"Executable labels are required before evaluation: {sorted(missing)}")
        if train_df.empty or test_df.empty:
            return {"status": "SKIPPED", "reason": "Insufficient data after feature construction"}
        if not (train_df["label_end_timestamp"] < test_df["timestamp"].iloc[0]).all():
            raise ValueError("Training-label horizon overlaps the test period")

        feature_cols = self._feature_columns(train_df)
        if set(feature_cols) - set(test_df.columns):
            raise ValueError("Test fold does not contain the training feature schema")
        y_train = train_df["label"].astype(int)
        if y_train.nunique() < 2:
            return {"status": "SKIPPED", "reason": "Training fold contains fewer than two action classes"}

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(train_df[feature_cols], y_train)
        prediction = model.predict(test_df[feature_cols]).astype(int)
        truth = test_df["label"].astype(int).to_numpy()

        gross_returns = np.select(
            [prediction == 1, prediction == -1],
            [test_df["long_return_h"].to_numpy(), test_df["short_return_h"].to_numpy()],
            default=0.0,
        )
        traded = prediction != 0
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth,
            prediction,
            labels=[-1, 0, 1],
            average="macro",
            zero_division=0,
        )
        metrics = {
            "status": "SUCCESS",
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "feature_count": len(feature_cols),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "trade_coverage": float(np.mean(traded)),
            "mean_executable_return_per_bar": float(np.mean(gross_returns)),
            "mean_executable_return_per_trade": float(np.mean(gross_returns[traded])) if traded.any() else 0.0,
            "positive_trade_fraction": float(np.mean(gross_returns[traded] > 0.0)) if traded.any() else 0.0,
            "cumulative_executable_return": float(np.prod(1.0 + gross_returns) - 1.0),
            "model": model,
        }
        logger.info(
            "Baseline fold evaluation completed",
            **{key: value for key, value in metrics.items() if key != "model"},
        )
        return metrics

    def evaluate_purged_walk_forward(
        self,
        labelled_df: pd.DataFrame,
        pipeline: LeakageSafeFeaturePipeline,
        n_splits: int = 5,
        purge_bars: int | None = None,
        embargo_bars: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate a baseline model over independent chronological test folds."""
        folds = list(
            pipeline.iter_purged_walk_forward_folds(
                labelled_df,
                n_splits=n_splits,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
            )
        )
        if not folds:
            return {"status": "SKIPPED", "reason": "No valid chronological folds"}

        evaluations: List[Dict[str, Any]] = []
        for fold in folds:
            train = labelled_df.iloc[fold.train_start : fold.train_end_exclusive]
            test = labelled_df.iloc[fold.test_start : fold.test_end_exclusive]
            evaluation = self.train_baseline_classifier(train, test)
            evaluation["fold"] = asdict(fold)
            evaluation.pop("model", None)
            evaluations.append(evaluation)

        successful = [item for item in evaluations if item["status"] == "SUCCESS"]
        if not successful:
            return {"status": "SKIPPED", "reason": "All folds were skipped", "folds": evaluations}

        return {
            "status": "RESEARCH_ONLY",
            "folds": evaluations,
            "successful_folds": len(successful),
            "mean_balanced_accuracy": float(np.mean([item["balanced_accuracy"] for item in successful])),
            "mean_macro_f1": float(np.mean([item["macro_f1"] for item in successful])),
            "mean_trade_coverage": float(np.mean([item["trade_coverage"] for item in successful])),
            "mean_executable_return_per_bar": float(
                np.mean([item["mean_executable_return_per_bar"] for item in successful])
            ),
            "aggregate_cumulative_return": float(
                np.prod([1.0 + item["cumulative_executable_return"] for item in successful]) - 1.0
            ),
            "disclosure": (
                "These are historical, purged out-of-sample research metrics. They are not a "
                "prediction of future performance and do not authorize live trading."
            ),
        }
