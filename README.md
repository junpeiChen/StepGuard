# StepGuard testExample-26

`eval/testExample-26.py` is a quick test script for running the full StepGuard method on all supported datasets.

It evaluates the first 26 examples from each dataset:

- CovidQA
- DROP_History
- DROP_Nfl
- FinanceBench
- Halueval
- PubmedQA
- RAGTruth

This script does not run ablation. It uses the full StepGuard pipeline.

## Requirements

Install dependencies:

```bash
pip install torch transformers numpy scipy nltk rouge-score jinja2 requests tqdm
```

Prepare a local HuggingFace model and set its path with `STEPGUARD_MODEL_PATH`.

Windows PowerShell:

```powershell
$env:STEPGUARD_MODEL_PATH="D:\models\Qwen3-8B"
```

Linux / macOS:

```bash
export STEPGUARD_MODEL_PATH=/path/to/Qwen3-8B
```

## Run

From the project root:

```bash
python StepGuard-main/eval/testExample-26.py
```

Or from `StepGuard-main`:

```bash
cd StepGuard-main
python eval/testExample-26.py
```

## Outputs

Results are saved to:

```text
StepGuard-main/eval/testExample-26/
```

The final metric summary is saved as:

```text
StepGuard-main/eval/testExample-26/testExample-26_summary.json
```

Metrics include:

- EM
- F1
- BLEU
- ROUGE-L
- runtime

## Notes

- The script runs the full StepGuard method.
- It tests 26 examples for each dataset.
- It requires a local HuggingFace model.
- GPU is recommended for faster inference.
