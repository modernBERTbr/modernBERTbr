# ModernBertBR

## How to run

Install `uv`:
https://docs.astral.sh/uv/getting-started/installation/

Then, create a virtual environment and install the dependencies:

```bash
uv venv
uv sync
```

Run the script:

```bash
uv run main.py
```

## Notes

ModernBERT
epoch: 1.49
steps: 60000
loss: 1.596

## Token Distribution Statistics

File: brwac/brwac_plain.txt
Total lines analyzed: 50,000,000
Total tokens found: 1,223,495,027

Mean tokens per line: 24.47
Median tokens per line: 20.00
Min tokens per line: 0
Max tokens per line: 13885
Standard deviation: 22.11
90th percentile: 50.00
95th percentile: 62.00
99th percentile: 92.00

BERT reduced learning rate after gradient explosion:
5e-5

{'loss': 2.069, 'grad_norm': 1.6301755905151367, 'learning_rate': 2.7111540941328178e-05, 'epoch': 1.54}
75000 steps
