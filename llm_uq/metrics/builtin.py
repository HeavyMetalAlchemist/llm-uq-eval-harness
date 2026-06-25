from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .base import BaseMetric, MetricResult
from .registry import register_metric


def _clean(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (conf_array, labels_array) with NaNs removed."""
    mask = df[col].notnull() & df["correct"].notnull()
    conf = df.loc[mask, col].astype(float).values
    labels = df.loc[mask, "correct"].astype(float).values
    return conf, labels


@register_metric
class ConfidenceMetric(BaseMetric):
    name = "confidence"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        conf, labels = _clean(df, "confidence")
        return MetricResult(
            name=self.name,
            scalars={"accuracy": float(labels.mean()) if len(labels) else float("nan")},
        )


@register_metric
class BrierMetric(BaseMetric):
    name = "brier"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        conf, labels = _clean(df, "confidence")
        score = float(np.mean((conf - labels) ** 2)) if len(conf) else float("nan")
        return MetricResult(name=self.name, scalars={"brier_score": score})


@register_metric
class ECEMetric(BaseMetric):
    name = "ece"

    def compute(self, df: pd.DataFrame, n_bins: int = 10) -> MetricResult:
        conf, labels = _clean(df, "confidence")
        if len(conf) < 2:
            return MetricResult(name=self.name, scalars={"ece": float("nan")})

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        e = 0.0
        N = len(conf)
        bin_conf_list, bin_acc_list, bin_counts = [], [], []

        for i in range(n_bins):
            mask = (conf >= bins[i]) & (conf <= bins[i + 1]) if i == 0 else (conf > bins[i]) & (conf <= bins[i + 1])
            if not np.any(mask):
                continue
            acc = float(np.mean(labels[mask]))
            avg_c = float(np.mean(conf[mask]))
            cnt = int(np.sum(mask))
            e += (cnt / N) * abs(acc - avg_c)
            bin_conf_list.append(avg_c)
            bin_acc_list.append(acc)
            bin_counts.append(cnt)

        return MetricResult(
            name=self.name,
            scalars={"ece": float(e)},
            extras={"bin_conf": bin_conf_list, "bin_acc": bin_acc_list, "bin_counts": bin_counts},
        )


@register_metric
class AUROCMetric(BaseMetric):
    name = "auroc"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        methods = {
            "1 - confidence": 1.0 - pd.to_numeric(df.get("confidence"), errors="coerce"),
            "mean_token_entropy": pd.to_numeric(df.get("mean_token_entropy"), errors="coerce"),
            "std_logp": pd.to_numeric(df.get("std_logp"), errors="coerce"),
            "std_logp_last_k": pd.to_numeric(df.get("std_logp_last_k"), errors="coerce"),
            "max_token_entropy": pd.to_numeric(df.get("max_token_entropy"), errors="coerce"),
            "-min_token_margin": -pd.to_numeric(df.get("min_token_margin"), errors="coerce"),
            "1 - min_chosen_prob": 1.0 - pd.to_numeric(df.get("min_chosen_prob"), errors="coerce"),
        }
        y_err = 1.0 - df["correct"].astype(float).values
        scalars: dict[str, float] = {}
        for method_name, series in methods.items():
            if series is None:
                continue
            m = series.notnull() & np.isfinite(y_err)
            if m.sum() < 2 or len(np.unique(y_err[m])) < 2:
                continue
            scalars[f"auroc_{method_name}"] = float(roc_auc_score(y_err[m], series[m].values))
        return MetricResult(name=self.name, scalars=scalars)


@register_metric
class EntropyMetric(BaseMetric):
    name = "entropy"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        conf, labels = _clean(df, "mean_token_entropy")
        if len(conf) < 2 or len(np.unique(labels)) < 2:
            return MetricResult(name=self.name, scalars={})
        y_err = 1.0 - labels
        auroc = float(roc_auc_score(y_err, conf))
        return MetricResult(
            name=self.name,
            scalars={"mean_token_entropy_auroc": auroc},
        )


@register_metric
class StdLogpMetric(BaseMetric):
    name = "std_logp"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        conf, labels = _clean(df, "std_logp")
        if len(conf) < 2 or len(np.unique(labels)) < 2:
            return MetricResult(name=self.name, scalars={})
        y_err = 1.0 - labels
        auroc = float(roc_auc_score(y_err, conf))
        return MetricResult(name=self.name, scalars={"std_logp_auroc": auroc})


@register_metric
class RiskCoverageMetric(BaseMetric):
    name = "risk_coverage"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        from sklearn.metrics import auc  # noqa: PLC0415

        y = df["correct"].astype(float).values
        signals = {
            "1 - confidence": 1.0 - pd.to_numeric(df.get("confidence"), errors="coerce"),
            "mean_token_entropy": pd.to_numeric(df.get("mean_token_entropy"), errors="coerce"),
            "std_logp": pd.to_numeric(df.get("std_logp"), errors="coerce"),
        }
        coverages = np.linspace(0, 1, 100)
        scalars: dict[str, float] = {}
        extras: dict[str, list] = {}

        for sig_name, u in signals.items():
            if u is None:
                continue
            m = u.notnull() & np.isfinite(y)
            if m.sum() < 2 or len(np.unique(y[m])) < 2:
                continue
            order = np.argsort(u[m].values)
            y_sorted = y[m][order]
            risks = []
            for frac in coverages:
                k = int(frac * len(y_sorted))
                risks.append(0.0 if k == 0 else 1.0 - y_sorted[:k].mean())
            aurc = float(auc(coverages, risks))
            key = sig_name.replace(" ", "_").replace("-", "neg")
            scalars[f"aurc_{key}"] = aurc
            extras[sig_name] = {"coverages": coverages.tolist(), "risks": risks}

        return MetricResult(name=self.name, scalars=scalars, extras=extras)
