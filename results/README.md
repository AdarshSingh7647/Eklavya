# Results

Evaluation outputs for **Std-SFT** and **Eklav**.

```
results/
├── passage_reranking/
│   ├── summary.csv                      one row per run: model, method, run, avg and per-domain nDCG@10
│   └── <model>/<method>/bright/<run>/
│       ├── metrics.json                 all metrics, per domain, plus checkpoint and notes
│       └── rankings/<domain>.jsonl.xz   per query: the candidates with score and relevance label
└── table_reranking/
    ├── summary.csv                      one row per method and dataset
    └── qwen3-8b/<method>/<dataset>/
        ├── metrics.json                 nDCG, accuracy, parse failures, thinking tokens
        └── predictions.json.xz          per query: candidates, generated reasoning, final ranking
```

`<model>` is the base model (`qwen3-8b`, `qwen3-14b`, `glm-z1-9b`). `<method>` is `std-sft` or `eklav`. `<run>`: `run1` and `run2` are two evaluation runs of the same model.

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

## Passage reranking: BRIGHT, nDCG@10

Pointwise reranking of the BM25 top-100 candidates released with Rank1. Each row is the mean over
its runs. Domains: Bio = biology, Earth = earth_science, Econ = economics, Psy = psychology,
Rob = robotics, Stack = stackoverflow, Sus = sustainable_living, Leet = leetcode, AoPS = aops,
TQ-Q / TQ-T = theoremqa_questions / theoremqa_theorems.

| Model | Method | Runs | Avg | Bio | Earth | Econ | Psy | Rob | Stack | Sus | Leet | Pony | AoPS | TQ-Q | TQ-T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | std-sft | 2 | 31.5 | 45.2 | 43.1 | 29.3 | 35.8 | 24.2 | 30.5 | 34.9 | 21.6 | 33.4 | **9.7** | 27.0 | 43.4 |
| qwen3-8b | eklav | 2 | 34.2 | 50.0 | 49.2 | 30.9 | **42.4** | **27.9** | 32.4 | 35.6 | **25.7** | **37.6** | **9.7** | 28.5 | 41.0 |
| glm-z1-9b | eklav | 2 | 33.2 | 48.5 | 45.3 | 30.4 | 40.1 | 23.6 | 32.7 | **36.6** | 24.9 | 36.1 | 7.8 | **28.8** | 43.6 |
| qwen3-14b | std-sft | 1 | 32.5 | 46.6 | 43.6 | 31.7 | 42.1 | 26.1 | 31.5 | 35.6 | 19.4 | 37.1 | 9.2 | 25.7 | 41.3 |
| qwen3-14b | eklav | 1 | **34.7** | **50.6** | **50.0** | **32.7** | **42.4** | 27.4 | **33.7** | 35.5 | 24.7 | 36.1 | **9.7** | 28.3 | **44.7** |

## Table reranking: nDCG@10, Qwen3-8B

Listwise reranking with a sliding window (Eklav checkpoint-300, Std-SFT checkpoint-160). Further
metrics are in [table_reranking/summary.csv](table_reranking/summary.csv).

| Dataset | Queries | Method | nDCG@10 | Parse failures | Avg thinking tokens |
|---|---|---|---|---|---|
| sqa | 148 | std-sft | 72.05 | 0 | 3318 |
| sqa | 148 | eklav | **73.81** | 0 | 2145 |
| tatqa | 362 | std-sft | 52.03 | 23 | 3704 |
| tatqa | 362 | eklav | **54.59** | 0 | 2473 |
| tabfact | 15106 | std-sft | 66.61 | 583 | 3716 |
| tabfact | 15106 | eklav | **68.90** | 45 | 2454 |
| nq | 966 | std-sft | **74.73** | 27 | 2404 |
| nq | 966 | eklav | 72.20 | 22 | 1362 |
| aitqa | 515 | std-sft | **88.25** | 12 | 1660 |
| aitqa | 515 | eklav | 85.99 | 13 | 1418 |
