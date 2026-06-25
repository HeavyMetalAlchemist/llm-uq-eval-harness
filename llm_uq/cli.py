from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

app = typer.Typer(
    name="llm-uq",
    help="Token-level uncertainty quantification for HuggingFace causal LMs.",
    no_args_is_help=True,
)


def _load_config(config: Path):
    from .config import RunConfig
    return RunConfig.from_yaml(config)


def _normalize_label_col(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Return a copy of df with the label column aliased to 'correct' for metrics."""
    if label_col == "correct":
        return df
    df = df.copy()
    df["correct"] = df[label_col]
    return df


def _run_metrics(df: pd.DataFrame, cfg) -> list:
    from .metrics.registry import get_metric, load_builtins
    load_builtins()
    results = []
    for name in cfg.metrics.names:
        try:
            metric = get_metric(name)
            result = metric.compute(df)
            results.append(result)
        except KeyError as e:
            typer.echo(f"Warning: {e}", err=True)
    return results


def _build_config_info(cfg) -> dict:
    return {
        "model": cfg.model.name,
        "quantize_4bit": cfg.model.quantize_4bit,
        "dataset": cfg.dataset.name,
        "n_samples": cfg.dataset.n_samples,
        "scoring": cfg.dataset.scoring,
        "system_prompt": (cfg.prompt.get_system() or "")[:80] + ("..." if len(cfg.prompt.get_system() or "") > 80 else ""),
    }


def _emit_report(df: pd.DataFrame, metric_results: list, cfg, label_col: str = "correct") -> None:
    from .report import build_report, build_markdown_report
    config_info = _build_config_info(cfg)
    out = cfg.output.path
    if cfg.output.format == "html":
        build_report(df, metric_results, config_info, label_col=label_col, output_path=out)
    else:
        build_markdown_report(df, metric_results, config_info, label_col=label_col, output_path=out)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
):
    """Run inference + compute metrics + generate report."""
    cfg = _load_config(config)
    typer.echo(f"Loading model: {cfg.model.name}")

    from .engine import ModelLoader, generate_response
    loader = ModelLoader(cfg.model).load()
    model, tok = loader.model, loader.tok

    from .datasets.loader import iter_examples
    from .datasets.scoring import score

    math_stopping = cfg.dataset.scoring == "numeric"
    rows = []
    examples = list(iter_examples(cfg.dataset))
    typer.echo(f"Running inference on {len(examples)} examples...")

    for i, ex in enumerate(examples, 1):
        out = generate_response(
            question=ex["input"],
            model=model,
            tok=tok,
            prompt_cfg=cfg.prompt,
            model_cfg=cfg.model,
            math_stopping=math_stopping,
        )
        correct = score(out["gen_text"], ex["gold"], cfg.dataset.scoring, cfg.dataset.cosine_threshold)
        row = {"id": ex["_idx"], "input": ex["input"], "gold": str(ex["gold"]), "correct": correct, **out}
        rows.append(row)
        if i % 10 == 0:
            typer.echo(f"  [{i}/{len(examples)}] done")

    df = pd.DataFrame(rows)

    if cfg.output.save_csv:
        csv_path = cfg.output.path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        typer.echo(f"Results CSV saved → {csv_path}")

    metric_results = _run_metrics(df, cfg)

    # determine label column (always 'correct' from our pipeline)
    _emit_report(df, metric_results, cfg, label_col="correct")


@app.command()
def eval(
    results: Path = typer.Option(..., "--results", "-r", help="Path to results CSV."),
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
    label_col: str = typer.Option("correct", "--label-col", help="Column name for correctness labels."),
):
    """Compute metrics and generate report from an existing results CSV. No GPU needed."""
    cfg = _load_config(config)
    typer.echo(f"Loading results from {results}")
    df = pd.read_csv(results)

    if label_col not in df.columns:
        typer.echo(f"Error: label column '{label_col}' not found in CSV. Columns: {list(df.columns)}", err=True)
        raise typer.Exit(1)

    df_metrics = _normalize_label_col(df, label_col)
    metric_results = _run_metrics(df_metrics, cfg)

    # Print scalar summary to terminal
    for r in metric_results:
        for k, v in r.scalars.items():
            typer.echo(f"  {r.name}/{k}: {v:.4f}")

    _emit_report(df_metrics, metric_results, cfg, label_col=label_col)


@app.command()
def report(
    results: Path = typer.Option(..., "--results", "-r", help="Path to results CSV."),
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
    label_col: str = typer.Option("correct", "--label-col", help="Column name for correctness labels."),
):
    """Generate report only from an existing results CSV. No metrics recomputed."""
    cfg = _load_config(config)
    typer.echo(f"Loading results from {results}")
    df = pd.read_csv(results)

    if label_col not in df.columns:
        typer.echo(f"Error: label column '{label_col}' not found. Columns: {list(df.columns)}", err=True)
        raise typer.Exit(1)

    df = _normalize_label_col(df, label_col)
    _emit_report(df, [], cfg, label_col=label_col)
