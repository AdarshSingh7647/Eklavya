<div align="center">

# Eklav

**Supervise the answer, not the reasoning.**

Eklav puts a teacher's reasoning trace in the *input* of a training sequence and
trains the model on the *final answer only*. Passage and table reranking.

[![Models](https://img.shields.io/badge/🤗_Models-Hugging_Face-yellow)](https://huggingface.co/AdarshSingh7647)
[![Results](https://img.shields.io/badge/Results-browse-blue)](results/README.md)
![Python](https://img.shields.io/badge/python-3.9%2B-informational)

</div>

|  | Teacher trace is | Loss on |
|---|---|---|
| **Std-SFT** | the response (imitation target) | trace + answer |
| **Eklav** | a hint in the prompt | answer only |

## 🤗 Models

| Task | Base model | Std-SFT | Eklav |
|---|---|---|---|
| Passage reranking | Qwen3-0.6B | [Eklav-0.6B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-0.6B-Reranker-CotGen) | [Eklav-0.6B-Reranker](https://huggingface.co/AdarshSingh7647/Eklav-0.6B-Reranker) |
| | Qwen3-4B | [Eklav-4B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-4B-Reranker-CotGen) | [Eklav-4B-Reranker](https://huggingface.co/AdarshSingh7647/Eklav-4B-Reranker) |
| | Qwen2.5-7B | [Eklav-7B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-7B-Reranker-CotGen) | – |
| | Qwen3-8B | [Eklav-8B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-8B-Reranker-CotGen) | [Eklav-8B-Reranker](https://huggingface.co/AdarshSingh7647/Eklav-8B-Reranker) |
| | GLM-Z1-9B | [Eklav-9B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-9B-Reranker-CotGen) | [Eklav-9B-Reranker](https://huggingface.co/AdarshSingh7647/Eklav-9B-Reranker) |
| | Qwen3-14B | [Eklav-14B-Reranker-CotGen](https://huggingface.co/AdarshSingh7647/Eklav-14B-Reranker-CotGen) | [Eklav-14B-Reranker](https://huggingface.co/AdarshSingh7647/Eklav-14B-Reranker) |
| Table reranking | Qwen3-8B | [TabRankStandardSFT](https://huggingface.co/AdarshSingh7647/TabRankStandardSFT) | [TabRank](https://huggingface.co/AdarshSingh7647/TabRank) |

`eklav models` prints this list; `eklav download --task passage_reranking --method eklav --size 8b` fetches one.

## Results

nDCG@10, one evaluation run per cell (best of each pair in bold). Per-domain scores, repeat runs and all
generations are in [results/](results/README.md).

| BRIGHT (12-domain avg) | Std-SFT | Eklav |
|---|---|---|
| Qwen3-0.6B | **16.1** | 14.4 |
| Qwen3-4B | 29.1 | **29.9** |
| Qwen3-8B | 31.5 | **34.2** |
| GLM-Z1-9B | – | 33.2 |
| Qwen3-14B | 32.5 | **34.7** |

| Table reranking (Qwen3-8B) | Std-SFT | Eklav |
|---|---|---|
| aitqa | **88.2** | 86.0 |
| hybridqa | 75.2 | **79.2** |
| nq | **74.7** | 72.2 |
| sqa | 72.0 | **73.8** |
| tabfact | 66.6 | **68.9** |
| tatqa | 52.0 | **54.6** |
| wtq | 62.0 | **65.0** |

## Reproduce

Install with `pip install -e ".[train,eval]"`. Then train any task with either method
(`std-sft` or `eklav`) on any base model, using the task's raw corpus (the
[Rank1 training set](https://huggingface.co/datasets/jhu-clsp/rank1-training-data) for passage reranking, a teacher-trace
`sft_examples.json` for tables); the sequences, the answer-only loss mask and the hyperparameters are handled for you:

```bash
eklav train --task passage_reranking --method eklav --base_model Qwen/Qwen3-8B \
    --raw_path rank1_training_data.arrow --output_dir runs/eklav
```

Evaluate a released checkpoint, or your own with `--model <path>`, on BRIGHT, the IBM table benchmarks, or any new
benchmark (`--benchmark local:<dir>` or `hf:<repo>`):

```bash
eklav eval --task passage_reranking --benchmark bright --method eklav --size 8b
```

`eklav <command> --help` lists every option.
