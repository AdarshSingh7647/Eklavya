# Results

Every number in the main [README](../README.md) comes from the files here. Only the two
methods compared in the paper, **Std-SFT** and **Eklav**, are included.

```
results/
├── passage_reranking/
│   ├── summary.csv                      one row per run: model, method, run, avg and per-domain nDCG@10
│   ├── <model>/<method>/bright/<run>/
│   │   ├── metrics.json                 all metrics, per domain, plus checkpoint and notes
│   │   └── rankings/<domain>.jsonl.xz   per query: the candidates with score and relevance label
│   └── reasoning_examples/<method>/<domain>.jsonl
│                                        sampled judgments with the model's generated reasoning
└── table_reranking/
    ├── summary.csv                      one row per method and dataset
    └── qwen3-8b/<method>/<dataset>/
        ├── metrics.json                 nDCG, accuracy, parse failures, thinking tokens
        └── predictions.json.xz          per query: candidates, generated reasoning, final ranking
```

`<model>` is the base model (`qwen3-0.6b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `glm-z1-9b`,
`qwen2.5-7b`). `<method>` is `std-sft` or `eklav`. `<run>`: `run1` is the run reported in
the main README; `run2` is a second evaluation run of the same model, except Qwen3-0.6B Eklav, where
`run2` was trained with a learning rate of 5e-5 instead of 1e-4.

## Reading the files

```python
import json, lzma

with lzma.open("results/passage_reranking/qwen3-8b/eklav/bright/run1/rankings/biology.jsonl.xz", "rt") as f:
    for line in f:
        query = json.loads(line)          # dataset, query_id, query, candidates[doc_id, score, label, passage]

with lzma.open("results/table_reranking/qwen3-8b/eklav/nq/predictions.json.xz", "rt") as f:
    predictions = json.load(f)            # query id -> question, raw_output, ranked_ids, parse_failed, ...
```

From a shell: `xzcat <file>.xz | head`.

## Passage reranking: BRIGHT, nDCG@10 (run1)

Pointwise reranking of the BM25 top-100 candidates released with Rank1. Domains:
Bio = biology, Earth = earth_science, Econ = economics, Psy = psychology, Rob = robotics,
Stack = stackoverflow, Sus = sustainable_living, Leet = leetcode, AoPS = aops,
TQ-Q / TQ-T = theoremqa_questions / theoremqa_theorems.

| Model | Method | Avg | Bio | Earth | Econ | Psy | Rob | Stack | Sus | Leet | Pony | AoPS | TQ-Q | TQ-T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b | std-sft | 16.1 | 26.0 | 17.0 | 13.3 | 24.4 | 13.6 | 18.4 | 17.7 | 12.9 | 13.8 | 3.9 | 15.1 | 16.6 |
| qwen3-0.6b | eklav | 14.4 | 9.6 | 12.1 | 11.6 | 20.8 | 11.6 | 18.9 | 13.9 | 13.9 | 16.9 | 2.4 | 17.6 | 23.7 |
| qwen3-4b | std-sft | 29.1 | 44.5 | 45.5 | 24.4 | 33.9 | 21.0 | 28.8 | 30.3 | 14.7 | 33.9 | 8.9 | 24.7 | 38.8 |
| qwen3-4b | eklav | 29.9 | 41.6 | 45.0 | 27.3 | 36.6 | 26.1 | 31.1 | 32.9 | 22.3 | 27.3 | 6.0 | 27.7 | 35.4 |
| qwen3-8b | std-sft | 31.5 | 45.5 | 43.1 | 29.3 | 35.8 | 24.2 | 30.5 | 34.9 | 21.6 | 33.4 | 9.7 | 27.1 | 43.4 |
| qwen3-8b | eklav | 34.2 | 50.0 | 49.2 | 30.9 | 42.4 | 27.9 | 32.4 | 35.6 | 25.7 | 37.5 | 9.7 | 28.5 | 41.0 |
| glm-z1-9b | eklav | 33.2 | 48.5 | 45.3 | 30.4 | 40.1 | 23.6 | 32.7 | 36.6 | 24.9 | 36.1 | 7.7 | 28.8 | 43.6 |
| qwen3-14b | std-sft | 32.5 | 46.6 | 43.6 | 31.7 | 42.1 | 26.1 | 31.5 | 35.6 | 19.4 | 37.1 | 9.2 | 25.7 | 41.3 |
| qwen3-14b | eklav | 34.7 | 50.6 | 50.0 | 32.7 | 42.4 | 27.4 | 33.7 | 35.5 | 24.7 | 36.1 | 9.7 | 28.3 | 44.7 |
| qwen2.5-7b | std-sft | 26.1 | 43.6 | 33.7 | 24.2 | 37.3 | 19.8 | 24.0 | 31.5 | 14.4 | 22.0 | 4.4 | 18.7 | 39.0 |

The training step of each run is recorded in its `metrics.json` (`null` where the original export did not record it).

## Table reranking: Qwen3-8B (Eklav checkpoint-300, Std-SFT checkpoint-160)

Listwise reranking with a sliding window. Full metrics (nDCG, accuracy, recall, precision,
parse failures, thinking tokens) are in [table_reranking/summary.csv](table_reranking/summary.csv).

| Dataset | Queries | Method | nDCG@10 | acc@1 | Parse failures | Avg thinking tokens |
|---|---|---|---|---|---|---|
| aitqa | 515 | std-sft | 88.2 | 19.2 | 12 | 1660 |
| aitqa | 515 | eklav | 86.0 | 19.8 | 13 | 1418 |
| hybridqa | 6106 | std-sft | 75.2 | 20.9 | 325 | 3605 |
| hybridqa | 6106 | eklav | 79.2 | 21.7 | 7 | 1990 |
| nq | 966 | std-sft | 74.7 | 57.7 | 27 | 2404 |
| nq | 966 | eklav | 72.2 | 56.3 | 22 | 1362 |
| sqa | 148 | std-sft | 72.0 | 16.9 | 0 | 3318 |
| sqa | 148 | eklav | 73.8 | 14.2 | 0 | 2145 |
| tabfact | 15106 | std-sft | 66.6 | 0.0 | 583 | 3716 |
| tabfact | 15106 | eklav | 68.9 | 0.0 | 45 | 2454 |
| tatqa | 362 | std-sft | 52.0 | 14.1 | 23 | 3704 |
| tatqa | 362 | eklav | 54.6 | 14.4 | 0 | 2473 |
| wtq | 2425 | std-sft | 62.0 | 27.5 | 171 | 3171 |
| wtq | 2425 | eklav | 65.0 | 28.1 | 84 | 2192 |
