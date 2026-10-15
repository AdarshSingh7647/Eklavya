# Eklav

**Reasoning-conditioned answer supervision for reranking.**
Eklav keeps a teacher's reasoning trace in the *input* of the training sequence and
supervises only the *final answer*. This repository trains and evaluates it, together
with the standard-SFT baseline, on **passage reranking** and **table reranking**.

[**Models**](#released-models) ·
[**Results**](#results) ·
[**Quickstart**](#quickstart) ·
[**Train**](#train) ·
[**Evaluate**](#evaluate) ·
[**Result files**](results/README.md) ·
[**Hugging Face page**](https://huggingface.co/AdarshSingh7647)

| | Teacher trace | Loss on |
|---|---|---|
| **Std-SFT** | in the response, as the imitation target | trace and answer |
| **Eklav** | in the prompt, as a hint | answer only |

## Quickstart

```bash
pip install -e ".[eval]"        # vLLM + retrieval stack for evaluation
pip install -e ".[train]"       # LLaMA-Factory for training

# Evaluate a released checkpoint on BRIGHT. No paths needed: the model is
# downloaded from the Hugging Face Hub.
eklav eval --task passage_reranking --benchmark bright --method eklav --size 8b
```

`eklav --help` lists every command: `train`, `eval`, `models`, `download`, `data`.

## Released models

All checkpoints are merged bf16 models on the Hugging Face Hub. `eklav models` prints
this table; `eklav download --task passage_reranking --method eklav --size 8b` fetches one.

**Passage reranking** (pointwise: the model reasons, then answers `true` / `false`)

| Base model | Std-SFT | Eklav |
|---|---|---|
| Qwen3-0.6B | [Eklav-0.6B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-0.6B-Reranker-CotGen) | [Eklav-0.6B-Reranker](https://huggingface.co/AdarshSingh7647/Eklav-0.6B-Reranker) |
| Qwen3-4B | [Eklav-4B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-4B-Reranker-CotGen) | [Eklav-4B-Reranker](https://huggingface.co/AdarshSingh7647/Eklav-4B-Reranker) |
| Qwen2.5-7B | [Eklav-7B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-7B-Reranker-CotGen) | – |
| Qwen3-8B | [Eklav-8B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-8B-Reranker-CotGen) | [Eklav-8B-Reranker](https://huggingface.co/AdarshSingh7647/Eklav-8B-Reranker) |
| GLM-Z1-9B | [Eklav-9B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-9B-Reranker-CotGen) | [Eklav-9B-Reranker](https://huggingface.co/AdarshSingh7647/Eklav-9B-Reranker) |
| Qwen3-14B | [Eklav-14B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-14B-Reranker-CotGen) | [Eklav-14B-Reranker](https://huggingface.co/AdarshSingh7647/Eklav-14B-Reranker) |

**Table reranking** (listwise: one call ranks a window of candidate tables)

| Base model | Std-SFT | Eklav |
|---|---|---|
| Qwen3-8B | [TabRankStandardSFT](https://huggingface.co/AdarshSingh7647/TabRankStandardSFT) | [TabRank](https://huggingface.co/AdarshSingh7647/TabRank) |

(The Hub names `-CotGen` and `TabRankStandardSFT` are the Std-SFT baselines.)

## Results

Evaluation outputs are in [results/](results/README.md): per task, model, method and
dataset: the metrics, the per-query rankings, and generated reasoning.

**Passage reranking: BRIGHT, nDCG@10 averaged over its 12 domains**

| Base model | Std-SFT | Eklav | Δ |
|---|---|---|---|
| Qwen3-0.6B | 16.1 | 14.4 | -1.7 |
| Qwen3-4B | 29.1 | 29.9 | +0.8 |
| Qwen3-8B | 31.5 | 34.2 | +2.7 |
| GLM-Z1-9B | – | 33.2 | – |
| Qwen3-14B | 32.5 | 34.7 | +2.2 |
| Qwen2.5-7B | 26.1 | – | – |

**Table reranking: Qwen3-8B, nDCG@10**

| Dataset | Queries | Std-SFT | Eklav | Δ |
|---|---|---|---|---|
| aitqa | 515 | 88.2 | 86.0 | -2.3 |
| hybridqa | 6106 | 75.2 | 79.2 | +4.0 |
| nq | 966 | 74.7 | 72.2 | -2.5 |
| sqa | 148 | 72.0 | 73.8 | +1.8 |
| tabfact | 15106 | 66.6 | 68.9 | +2.3 |
| tatqa | 362 | 52.0 | 54.6 | +2.6 |
| wtq | 2425 | 62.0 | 65.0 | +3.0 |

Each cell is a single evaluation run (`run1`). See [results/README.md](results/README.md)
for repeat runs, checkpoints and the full per-domain numbers.

## Train

```bash
# Passage reranking: the Rank1 training set (jhu-clsp/rank1-training-data)
eklav train --task passage_reranking --method eklav --base_model Qwen/Qwen3-8B \
    --raw_path /path/to/rank1_training_data.arrow --output_dir runs/passage_eklav

# Table reranking: a JSON file of teacher traces (sft_examples.json)
eklav train --task table_reranking --method std-sft --base_model Qwen/Qwen3-8B \
    --raw_path /path/to/sft_examples.json --output_dir runs/table_std_sft
```

`--method` is `std-sft` or `eklav`; the ablations `naive-sft` (no teacher trace) and
`mask-only` (Std-SFT's sequence with the loss on the answer only) are also available.
`--base_model` is any local checkpoint or Hub id. The training data is built from
`--raw_path` on first use and kept under `data/`.

`eklav train` writes a LLaMA-Factory config, then launches `llamafactory-cli train`
with the answer-only loss mask loaded in every process, multi-GPU workers included.
Add `--dry_run` to only write the config, `--set key=value` to override any config
entry, and `--num_processes N` for multi-GPU. All methods of a task share one
hyperparameter set: LoRA rank 32 / alpha 64, learning rate 1e-4, effective batch 128,
two epochs. `--per_device_train_batch_size` only changes memory use; gradient
accumulation is recomputed to keep the effective batch at 128.

## Evaluate

```bash
# a released checkpoint
eklav eval --task passage_reranking --benchmark bright --method std-sft --size 4b

# your own model: a merged checkpoint, or a base model plus a LoRA adapter
eklav eval --task passage_reranking --benchmark bright --model runs/passage_eklav/merged
eklav eval --task table_reranking --benchmark FeTaQARetrieval \
    --model Qwen/Qwen3-8B --lora_path runs/table_eklav
```

Results are written to `results/<task>/<model_tag>/<benchmark>/` (override with
`--out_dir`): one row per judged pair or query with the full prompt and generation,
plus summary metrics. `--num_questions N --seed S` evaluates a reproducible sample.

`eklav eval --list_benchmarks` shows the built-in benchmarks: **BRIGHT** (12 domains)
and the IBM table benchmarks **AITQA, FeTaQA, OTT-QA, OpenWikiTables, NQ-Tables**.
**A new benchmark needs no code** if it is in BEIR layout (`corpus.jsonl`,
`queries.jsonl`, `qrels/<split>.tsv`) or a Hub dataset with `default` (qrels), `queries`
and `corpus` configs:

```bash
eklav eval --task table_reranking --benchmark local:/data/my_benchmark --split test
eklav eval --task passage_reranking --benchmark hf:org/my-benchmark
```

Benchmarks that ship no candidate pools get one from BM25 (passage) or a
BM25 + SPLADE + mpnet fusion (table); `--first_stage` overrides it. To register a loader
in Python see [eklav/eval/benchmarks.py](eklav/eval/benchmarks.py).

## How the methods differ

An example is `(x, r, a)`: task input, teacher reasoning trace, answer. At evaluation
all teacher context is removed and the model generates its own reasoning before it
answers. A method is a serialized sequence plus a loss mask
([eklav/methods.py](eklav/methods.py)):

| Method | Sequence | Loss on |
|---|---|---|
| **Std-SFT** | `[x; r + a]` | trace and answer |
| **Eklav** | `[x + h(r); a]` | answer |
| NAIVE-SFT (ablation) | `[x; a]` | answer |
| MASK-ONLY (ablation) | `[x; r + a]` | answer |

`h(r)` is the hint taken from the trace. For **passage reranking** it is the first
half of the trace's sentences with verdict-bearing sentences removed, so the Eklav
sequence is shorter than Std-SFT's. For **table reranking** it is the whole trace.

The answer-only mask covers everything up to and including `</think>`; it is applied
by [eklav/lf_plugin](eklav/lf_plugin), which patches LLaMA-Factory's encoder.

## Extending

- **New task:** add one module in [eklav/tasks/](eklav/tasks/) defining a `TASK` (raw-data
  reader, hint, prompt, hyperparameters) and register it; every method and every command
  then works for it.
- **New benchmark:** see [Evaluate](#evaluate).

## Repository layout

```
eklav/
  cli.py            the `eklav` command
  models.py         released checkpoints on the Hugging Face Hub
  methods.py        training methods and their loss masks
  data.py           training-data building
  tasks/            passage_reranking, table_reranking
  train.py          eklav train
  eval/             eklav eval: benchmarks, retrieval, evaluators
  lf_plugin/        answer-only loss mask for LLaMA-Factory
results/            evaluation outputs (see results/README.md)
tests/              pytest suite (no GPU needed)
```
