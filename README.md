<div align="center">

# Eklav

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

**Passage reranking**

| | Qwen3-0.6B | Qwen3-4B | Qwen3-8B | GLM-Z1-9B | Qwen3-14B |
|---|---|---|---|---|---|
| Std-SFT | [🤗](https://huggingface.co/AdarshSingh7647/Std-SFT-Passage-0.6B) | [🤗](https://huggingface.co/AdarshSingh7647/Std-SFT-Passage-4B) | [🤗](https://huggingface.co/AdarshSingh7647/Std-SFT-Passage-8B) | [🤗](https://huggingface.co/AdarshSingh7647/Std-SFT-Passage-9B) | [🤗](https://huggingface.co/AdarshSingh7647/Std-SFT-Passage-14B) |
| Eklav | [🤗](https://huggingface.co/AdarshSingh7647/Eklav-Passage-0.6B) | [🤗](https://huggingface.co/AdarshSingh7647/Eklav-Passage-4B) | [🤗](https://huggingface.co/AdarshSingh7647/Eklav-Passage-8B) | [🤗](https://huggingface.co/AdarshSingh7647/Eklav-Passage-9B) | [🤗](https://huggingface.co/AdarshSingh7647/Eklav-Passage-14B) |

**Table reranking**

| | Qwen3-8B |
|---|---|
| Naive-SFT | [🤗](https://huggingface.co/AdarshSingh7647/Naive-SFT-Table-8B) |
| Std-SFT | [🤗](https://huggingface.co/AdarshSingh7647/Std-SFT-Table-8B) |
| Eklav | [🤗](https://huggingface.co/AdarshSingh7647/Eklav-Table-8B) |

`eklav models` lists them; `eklav download --task passage_reranking --method eklav --size 8b` fetches one.

## Results: nDCG@10

Details: [results/](results/README.md).

**Passage reranking, BRIGHT (12-domain average)**

| | Qwen3-4B | Qwen3-8B | GLM-Z1-9B | Qwen3-14B |
|---|---|---|---|---|
| Std-SFT | 29.3 | 31.5 | 30.1 | 32.5 |
| Eklav | **31.5** | **34.2** | **33.2** | **34.7** |

**Table reranking, Qwen3-8B**

| | aitqa | hybridqa | nq | sqa | tabfact | tatqa | wtq |
|---|---|---|---|---|---|---|---|
| Std-SFT | **88.2** | 75.2 | **74.7** | 72.0 | 66.6 | 52.0 | 62.0 |
| Eklav | 86.0 | **79.2** | 72.2 | **73.8** | **68.9** | **54.6** | **65.0** |

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
