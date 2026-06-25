# LLM Uncertainty Quantification Eval Harness

A config-driven CLI tool for **token-level uncertainty quantification** of HuggingFace causal language models. Point it at any model, any dataset, and a system prompt - it runs inference, computes calibration metrics, and generates a self-contained HTML report.

Built on top of the research documented in the companion notebook repo: [llm-uncertainty-quantification](https://github.com/HeavyMetalAlchemist/llm-uncertainty-quantification), which covers the full methodology and findings for Phi-3-Mini-4k-Instruct on GSM8K and TriviaQA.

---

## What it does

For each model response, the tool captures:

**Token-level signals**
- Entropy per token, max entropy spike and its position
- Top-1 vs top-2 probability margin (decisiveness), and its minimum position
- Chosen token probability, and its minimum position

**Sequence-level signals**
- Confidence - geometric mean of chosen token probabilities
- Mean token entropy across the full sequence
- Std of log-probabilities (global and last-k tokens)

**Calibration metrics** (computed against correctness labels)
- Accuracy, Brier score, ECE (Expected Calibration Error)
- AUROC - how well uncertainty separates correct from incorrect predictions
- Risk-coverage curves (selective prediction / abstention)

**Output:** a single self-contained HTML report with 9 embedded plots. No external images, no server required - open it in any browser.

---

## Results

Evaluated on **Phi-3-Mini-4k-Instruct (4-bit quantized)**.

### GSM8K - Math Reasoning (200 samples)

| Metric | Value |
|---|---|
| Accuracy | 0.780 |
| Brier Score | 0.169 |
| ECE | 0.119 |
| AUROC (1 − confidence) | 0.752 |
| AUROC (mean token entropy) | 0.748 |

Key finding: uncertainty signals rank errors well (AUROC ~0.75). Abstaining on the least confident 20% raises accuracy from 78% to 86%.

### TriviaQA - Factual QA (500 samples)

| Metric | Value |
|---|---|
| Accuracy (contain) | 0.430 |
| Brier Score | 0.245 |
| ECE | 0.262 |
| AUROC (1 − confidence) | 0.834 |
| AUROC (mean token entropy) | 0.836 |

Key finding: strong error detection (AUROC ~0.83) but poor calibration (ECE ~0.26). The model is overconfident on factual recall - uncertainty ranks errors well but confidence scores cannot be trusted as probabilities.

See the full analysis in [`docs/gsm8k_report.html`](docs/gsm8k_report.html) and [`docs/triviaqa_report.html`](docs/triviaqa_report.html).

---

## Installation

**CPU only** (metrics + report generation, no inference):
```bash
pip install .
```

**GPU** (full inference pipeline, requires CUDA):
```bash
pip install ".[gpu]"
```

---

## Usage

### Run inference + metrics + report
Requires a CUDA GPU.
```bash
llm-uq run --config configs/gsm8k_example.yaml
```

### Compute metrics + report from an existing CSV
No GPU needed. Use this to iterate on metrics without re-running inference.
```bash
llm-uq eval --results results.csv --config configs/gsm8k_example.yaml
```
If your correctness column has a different name (e.g. TriviaQA uses `contain`):
```bash
llm-uq eval --results triviaqa_phi3_500.csv --config configs/triviaqa_example.yaml --label-col contain
```

### Regenerate report only
```bash
llm-uq report --results results.csv --config configs/gsm8k_example.yaml
```

---

## Configuration

Everything is controlled via a single YAML file. No code changes needed to run on a new model or dataset.

```yaml
model:
  name: microsoft/Phi-3-mini-4k-instruct   # any HuggingFace causal LM
  quantize_4bit: true                        # set false for CPU / non-CUDA
  device_map: auto
  max_new_tokens: 512

dataset:
  source: huggingface        # or: csv
  name: gsm8k
  hf_config: main            # HuggingFace dataset config name if required
  split: "test[:500]"
  input_col: question        # column used as the user prompt
  output_col: answer         # column with the ground truth
  n_samples: 200
  seed: 42
  scoring: numeric           # numeric | exact_match | contains | f1 | cosine

prompt:
  system: |
    You are a careful math solver. Output FINAL: <number> then END.

metrics:
  - confidence
  - brier
  - ece
  - auroc
  - entropy
  - std_logp
  - risk_coverage

output:
  format: html               # or: markdown
  path: ./report.html
  save_csv: true
```

### Scoring modes

| Mode | Use case |
|---|---|
| `numeric` | Math / calculation tasks - extracts the last number from output |
| `exact_match` | Strict span QA - normalized string equality |
| `contains` | Flexible span QA - gold answer appears anywhere in prediction |
| `f1` | Token-level F1 against best gold alias |
| `cosine` | Semantic similarity via sentence-transformers |

### Using your own dataset

Provide a CSV with at least an input column and a gold answer column:

```yaml
dataset:
  source: csv
  name: ./my_dataset.csv
  input_col: question
  output_col: gold_answer
  scoring: contains

prompt:
  system: |
    Answer concisely with only the key fact.
```

See [`configs/custom_example.yaml`](configs/custom_example.yaml) for a full template.

---

## Adding a custom metric (plugin interface)

Implement `BaseMetric` and register it - no changes to core code required:

```python
from llm_uq.metrics.base import BaseMetric, MetricResult
from llm_uq.metrics.registry import register_metric
import pandas as pd

@register_metric
class MyMetric(BaseMetric):
    name = "my_metric"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        score = float(df["confidence"].mean())
        return MetricResult(name=self.name, scalars={"mean_confidence": score})
```

Then add `my_metric` to the `metrics` list in your config.

---

## Project structure

```
llm_uq/
├── cli.py           # Typer CLI - run, eval, report subcommands
├── config.py        # Pydantic config models, YAML loading
├── engine.py        # Model loading + chunked inference with token stats
├── datasets/
│   ├── loader.py    # HuggingFace or CSV dataset loader
│   └── scoring.py   # Correctness scoring: numeric, exact_match, contains, f1, cosine
├── metrics/
│   ├── base.py      # BaseMetric ABC - plugin interface
│   ├── builtin.py   # Built-in metrics: brier, ece, auroc, entropy, std_logp, risk_coverage
│   └── registry.py  # register_metric(), get_metric()
└── report.py        # 9 plot functions + self-contained HTML report builder

configs/
├── gsm8k_example.yaml
├── triviaqa_example.yaml
└── custom_example.yaml

docs/
├── gsm8k_report.html      # Full calibration report, Phi-3 on GSM8K
└── triviaqa_report.html   # Full calibration report, Phi-3 on TriviaQA
```

---

## Recommended workflow for GPU instances

Inference is expensive. The pipeline is designed so you only run it once:

```bash
# On EC2 g4dn (or any CUDA machine)
pip install ".[gpu]"
llm-uq run --config configs/gsm8k_example.yaml   # saves results.csv + report.html

# Pull results.csv back locally, iterate on metrics/report without re-running inference
llm-uq eval --results results.csv --config configs/gsm8k_example.yaml
```

---

## Research notebook

The companion repo [llm-uncertainty-quantification](https://github.com/HeavyMetalAlchemist/llm-uncertainty-quantification) contains the full research documentation: methodology, dataset analysis, per-metric breakdowns, case studies with per-token entropy heatmaps, and cross-dataset conclusions. The notebook is the research proof; this tool is the engineering layer on top of it.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- For inference: CUDA GPU, `pip install ".[gpu]"` (adds `bitsandbytes` for 4-bit quantization)
- All other dependencies installed automatically via `pip install .`
