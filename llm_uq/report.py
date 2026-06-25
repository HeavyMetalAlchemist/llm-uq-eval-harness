from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.metrics import auc, roc_auc_score, roc_curve

from .metrics.base import MetricResult

# ── plot helpers ──────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;">'


# ── individual plot functions ─────────────────────────────────────────────────

def plot_reliability(df: pd.DataFrame, conf_col: str = "confidence", label_col: str = "correct", n_bins: int = 10) -> str:
    x = pd.to_numeric(df[conf_col], errors="coerce")
    y = pd.to_numeric(df[label_col], errors="coerce")
    mask = x.notnull() & y.notnull()
    x, y = x[mask].values, y[mask].values
    if x.size == 0:
        return ""

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(x, bins, right=True)
    bin_conf, bin_acc, counts = [], [], []
    for b in range(1, len(bins)):
        m = idx == b
        if not np.any(m):
            continue
        bin_conf.append(float(x[m].mean()))
        bin_acc.append(float(y[m].mean()))
        counts.append(int(m.sum()))

    if not counts:
        return ""
    bin_conf = np.array(bin_conf); bin_acc = np.array(bin_acc); counts = np.array(counts)
    ece = float(np.sum((counts / counts.sum()) * np.abs(bin_acc - bin_conf)))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Ideal")
    ax.plot(bin_conf, bin_acc, "o-", lw=2, label="Empirical")
    ax.set_xlabel("Mean confidence (per bin)"); ax.set_ylabel("Accuracy (per bin)")
    ax.set_title(f"Reliability Diagram  |  ECE={ece:.3f}"); ax.legend()
    ax2 = ax.twinx()
    ax2.bar(bin_conf, counts, width=0.06, alpha=0.25, color="tab:blue", label="Bin count")
    ax2.set_ylabel("Count"); ax2.legend(loc="lower right")
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_auroc_bar(df: pd.DataFrame, label_col: str = "correct") -> str:
    methods = {
        "1 - confidence": 1.0 - pd.to_numeric(df.get("confidence"), errors="coerce"),
        "mean_token_entropy": pd.to_numeric(df.get("mean_token_entropy"), errors="coerce"),
        "std_logp (global)": pd.to_numeric(df.get("std_logp"), errors="coerce"),
        "std_logp_last_k": pd.to_numeric(df.get("std_logp_last_k"), errors="coerce"),
        "max_token_entropy": pd.to_numeric(df.get("max_token_entropy"), errors="coerce"),
        "-min_token_margin": -pd.to_numeric(df.get("min_token_margin"), errors="coerce"),
        "1 - min_chosen_prob": 1.0 - pd.to_numeric(df.get("min_chosen_prob"), errors="coerce"),
    }
    y = pd.to_numeric(df[label_col], errors="coerce").astype(float).values
    y_err = 1.0 - y
    names, aucs_list = [], []
    for name, series in methods.items():
        if series is None: continue
        m = series.notnull() & np.isfinite(y_err)
        if m.sum() < 2 or len(np.unique(y_err[m])) < 2: continue
        names.append(name)
        aucs_list.append(float(roc_auc_score(y_err[m], series[m].values)))
    if not aucs_list:
        return ""
    order = np.argsort(aucs_list)[::-1]
    names_s = [names[i] for i in order]; aucs_s = [aucs_list[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(names_s)), aucs_s, alpha=0.85)
    ax.set_xticks(range(len(names_s))); ax.set_xticklabels(names_s, rotation=30, ha="right")
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC (errors as positives)")
    ax.set_title("AUROC by Uncertainty Method")
    for i, v in enumerate(aucs_s):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_roc_curves(df: pd.DataFrame, label_col: str = "correct") -> str:
    y = pd.to_numeric(df[label_col], errors="coerce").astype(float).values
    y_err = 1.0 - y
    signals = {
        "1 - confidence": 1.0 - pd.to_numeric(df.get("confidence"), errors="coerce"),
        "mean_token_entropy": pd.to_numeric(df.get("mean_token_entropy"), errors="coerce"),
        "std_logp": pd.to_numeric(df.get("std_logp"), errors="coerce"),
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    for name, series in signals.items():
        if series is None: continue
        m = series.notnull() & np.isfinite(y_err)
        if m.sum() < 2 or len(np.unique(y_err[m])) < 2: continue
        fpr, tpr, _ = roc_curve(y_err[m], series[m].values)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Error Detection ROC")
    ax.legend(loc="lower right"); plt.tight_layout()
    return _fig_to_b64(fig)


def plot_hist_box(df: pd.DataFrame, feature_col: str = "mean_token_entropy", label_col: str = "correct") -> str:
    d = df[[feature_col, label_col]].copy()
    d[feature_col] = pd.to_numeric(d[feature_col], errors="coerce")
    d[label_col] = pd.to_numeric(d[label_col], errors="coerce")
    d = d.dropna()
    if len(d) == 0: return ""
    ok = d[label_col].astype(int) == 1
    vals_ok = d.loc[ok, feature_col].values
    vals_ng = d.loc[~ok, feature_col].values
    edges = np.histogram_bin_edges(d[feature_col].values, bins=20)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))
    ax1.hist(vals_ok, bins=edges, alpha=0.6, label="correct")
    ax1.hist(vals_ng, bins=edges, alpha=0.6, label="incorrect")
    ax1.set_xlabel(feature_col); ax1.set_ylabel("count")
    ax1.set_title(f"{feature_col}: correct vs incorrect"); ax1.legend()
    d_box = d.copy(); d_box[label_col] = d_box[label_col].astype(int)
    sns.boxplot(data=d_box, x=label_col, y=feature_col, ax=ax2)
    ax2.set_xlabel(f"{label_col} (0=wrong 1=right)"); ax2.set_ylabel(feature_col)
    ax2.set_title(f"{feature_col} by correctness")
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_risk_coverage(df: pd.DataFrame, label_col: str = "correct") -> str:
    y = df[label_col].astype(float).values
    signals = {
        "1 - confidence": 1.0 - pd.to_numeric(df.get("confidence"), errors="coerce"),
        "mean_token_entropy": pd.to_numeric(df.get("mean_token_entropy"), errors="coerce"),
        "std_logp": pd.to_numeric(df.get("std_logp"), errors="coerce"),
    }
    coverages = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, u in signals.items():
        if u is None: continue
        m = u.notnull() & np.isfinite(y)
        if m.sum() < 2 or len(np.unique(y[m])) < 2: continue
        order = np.argsort(u[m].values)
        y_s = y[m][order]
        risks = [0.0 if int(f * len(y_s)) == 0 else 1.0 - y_s[:int(f * len(y_s))].mean() for f in coverages]
        ax.plot(coverages, risks, label=f"{name} (AURC={auc(coverages, risks):.3f})")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Risk (1 - accuracy)")
    ax.set_title("Risk-Coverage Curve"); ax.legend(); plt.tight_layout()
    return _fig_to_b64(fig)


def plot_feature_box(df: pd.DataFrame, label_col: str = "correct") -> str:
    features = [f for f in ["confidence", "mean_token_entropy", "std_logp"] if f in df.columns]
    d = df[[label_col] + features].dropna().copy()
    if len(d) == 0: return ""
    d[label_col] = d[label_col].astype(int)
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 4))
    if len(features) == 1: axes = [axes]
    for ax, feat in zip(axes, features):
        sns.boxplot(data=d, x=label_col, y=feat, ax=ax)
        ax.set_xlabel("Correct (0=wrong 1=right)"); ax.set_title(feat)
    plt.suptitle("Feature Distributions by Correctness"); plt.tight_layout()
    return _fig_to_b64(fig)


def plot_scatter_features(df: pd.DataFrame, label_col: str = "correct") -> str:
    pairs = [("confidence", "mean_token_entropy"), ("confidence", "std_logp")]
    pairs = [(x, y) for x, y in pairs if x in df.columns and y in df.columns]
    if not pairs: return ""
    needed = [label_col] + list(dict.fromkeys(c for p in pairs for c in p))
    d = df[needed].dropna().copy(); d[label_col] = d[label_col].astype(int)
    palette = {0: "red", 1: "blue"}
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1: axes = [axes]
    for ax, (x, y) in zip(axes, pairs):
        sns.scatterplot(data=d, x=x, y=y, hue=label_col, palette=palette, alpha=0.6, s=40, ax=ax)
        ax.set_title(f"{x} vs {y}")
    plt.suptitle("Uncertainty Feature Scatter"); plt.tight_layout()
    return _fig_to_b64(fig)


def plot_length_effects(df: pd.DataFrame, label_col: str = "correct") -> str:
    needed = ["n_gen_tokens", "confidence", "mean_token_entropy", "std_logp", label_col]
    missing = [c for c in needed if c not in df.columns]
    if missing: return ""
    d = df[needed].dropna()
    if len(d) == 0: return ""
    colors = d[label_col].map({1: "blue", 0: "red"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, feat, title in zip(
        axes,
        ["confidence", "mean_token_entropy", "std_logp"],
        ["Confidence vs Length", "Entropy vs Length", "Std Log-Prob vs Length"],
    ):
        ax.scatter(d["n_gen_tokens"], d[feat], c=colors, alpha=0.6, s=30)
        ax.set_xlabel("n_gen_tokens"); ax.set_ylabel(feat)
        ax.set_title(title)
    plt.suptitle("Length Effects on Uncertainty Signals"); plt.tight_layout()
    return _fig_to_b64(fig)


def plot_position_uncertainty(df: pd.DataFrame, label_col: str = "correct") -> str:
    needed = ["argmax_token_entropy", "argmin_token_margin", "n_gen_tokens", label_col]
    missing = [c for c in needed if c not in df.columns]
    if missing: return ""
    d = df[needed].dropna().copy()
    if len(d) == 0: return ""
    d["pos_entropy_norm"] = d["argmax_token_entropy"] / d["n_gen_tokens"]
    d["pos_margin_norm"] = d["argmin_token_margin"] / d["n_gen_tokens"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(
        axes,
        ["pos_entropy_norm", "pos_margin_norm"],
        ["Entropy Peak Positions", "Margin Dip Positions"],
    ):
        ax.hist(
            [d.loc[d[label_col] == 1, col], d.loc[d[label_col] == 0, col]],
            bins=20, label=["correct", "incorrect"], alpha=0.6,
        )
        ax.set_xlabel("Relative position"); ax.set_ylabel("Count"); ax.set_title(title); ax.legend()
    plt.suptitle("Position-Aware Uncertainty"); plt.tight_layout()
    return _fig_to_b64(fig)


# ── HTML report builder ────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LLM Uncertainty Quantification Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto; padding: 2rem; color: #222; }}
  h1 {{ border-bottom: 2px solid #444; padding-bottom: .5rem; }}
  h2 {{ color: #333; margin-top: 2.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: .5rem 1rem; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; margin: 1.5rem 0; }}
  .plot-card {{ border: 1px solid #ddd; border-radius: 6px; padding: 1rem; }}
  .plot-card h3 {{ margin: 0 0 .75rem; font-size: 1rem; }}
  img {{ border-radius: 4px; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: .9em; }}
</style>
</head>
<body>
<h1>LLM Uncertainty Quantification Report</h1>

<h2>Run Configuration</h2>
{config_table}

<h2>Summary Metrics</h2>
{metrics_table}

<h2>Visualizations</h2>
<div class="plot-grid">
{plot_cards}
</div>

</body>
</html>
"""


def _dict_table(d: dict) -> str:
    rows = "".join(f"<tr><td><code>{k}</code></td><td>{v}</td></tr>" for k, v in d.items())
    return f"<table><tr><th>Key</th><th>Value</th></tr>{rows}</table>"


def _metrics_table(results: list[MetricResult]) -> str:
    rows = ""
    for r in results:
        for k, v in r.scalars.items():
            rows += f"<tr><td>{r.name}</td><td>{k}</td><td>{v:.4f}</td></tr>"
    return f"<table><tr><th>Metric</th><th>Key</th><th>Value</th></tr>{rows}</table>"


def _plot_card(title: str, b64: str) -> str:
    if not b64:
        return ""
    return f'<div class="plot-card"><h3>{title}</h3>{_img_tag(b64, title)}</div>'


def build_report(
    df: pd.DataFrame,
    metric_results: list[MetricResult],
    config_info: dict[str, Any],
    label_col: str = "correct",
    output_path: Path = Path("report.html"),
) -> None:
    config_table = _dict_table(config_info)
    metrics_table = _metrics_table(metric_results)

    plots = [
        ("Reliability Diagram", plot_reliability(df, label_col=label_col)),
        ("AUROC by Method", plot_auroc_bar(df, label_col=label_col)),
        ("Error Detection ROC", plot_roc_curves(df, label_col=label_col)),
        ("Mean Token Entropy vs Correctness", plot_hist_box(df, "mean_token_entropy", label_col=label_col)),
        ("Risk-Coverage Curve", plot_risk_coverage(df, label_col=label_col)),
        ("Feature Distributions by Correctness", plot_feature_box(df, label_col=label_col)),
        ("Uncertainty Feature Scatter", plot_scatter_features(df, label_col=label_col)),
        ("Length Effects on Uncertainty", plot_length_effects(df, label_col=label_col)),
        ("Position-Aware Uncertainty", plot_position_uncertainty(df, label_col=label_col)),
    ]

    plot_cards = "\n".join(_plot_card(title, b64) for title, b64 in plots if b64)
    html = _HTML_TEMPLATE.format(
        config_table=config_table,
        metrics_table=metrics_table,
        plot_cards=plot_cards,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved -> {output_path}")


def build_markdown_report(
    df: pd.DataFrame,
    metric_results: list[MetricResult],
    config_info: dict[str, Any],
    label_col: str = "correct",
    output_path: Path = Path("report.md"),
) -> None:
    lines = ["# LLM Uncertainty Quantification Report\n"]
    lines.append("## Run Configuration\n")
    for k, v in config_info.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("\n## Summary Metrics\n")
    lines.append("| Metric | Key | Value |")
    lines.append("|---|---|---|")
    for r in metric_results:
        for k, v in r.scalars.items():
            lines.append(f"| {r.name} | {k} | {v:.4f} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved -> {output_path}")
