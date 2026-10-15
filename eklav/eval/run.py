"""
Evaluate a model on a benchmark.

    # a released checkpoint (downloaded from the Hugging Face Hub on first use)
    eklav eval --task passage_reranking --benchmark bright --method eklav --size 8b

    # your own model: a merged checkpoint, or a base model plus a LoRA adapter
    eklav eval --task passage_reranking --benchmark bright --model runs/passage_eklav/merged
    eklav eval --task table_reranking --benchmark FeTaQARetrieval --model Qwen/Qwen3-8B \
        --lora_path runs/table_eklav

--benchmark is a registered name (`eklav eval --list_benchmarks`), `local:<dir>`
(BEIR layout) or `hf:<repo>`. See eklav.eval.benchmarks for adding your own.

Without --model the released checkpoint for --task / --method / --size is used (`eklav models`).
Results go to --out_dir (default results/<task>/<model_tag>/<benchmark>). Per subset it gets <subset>.jsonl (one row per judged pair or per
query, with the full prompt and raw generation), <subset>_summary.json, and a
summary.json covering the whole run.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

from eklav import models
from eklav.eval import benchmarks
from eklav.eval.model import ModelArgs

TASK_EVALUATORS = ("passage_reranking", "table_reranking")


def make_evaluator(task: str, model_args: ModelArgs, **options):
    if task == "passage_reranking":
        from eklav.eval.passage_reranking import PassageRerankingEvaluator as Evaluator
    elif task == "table_reranking":
        from eklav.eval.table_reranking import TableRerankingEvaluator as Evaluator
    else:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASK_EVALUATORS}")
    return Evaluator(model_args, **{k: v for k, v in options.items() if v is not None})


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def average(summaries: Dict[str, Dict]) -> Dict[str, float]:
    keys = [k for k, v in next(iter(summaries.values())).items() if isinstance(v, float)]
    return {k: round(sum(s[k] for s in summaries.values()) / len(summaries), 4) for k in keys}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="eklav eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASK_EVALUATORS)
    ap.add_argument("--benchmark", help="Registered name, local:<dir> or hf:<repo>.")
    ap.add_argument("--subsets", nargs="*", default=None, help="Subset of the benchmark's subsets (default: all).")
    ap.add_argument("--split", default="test", help="Qrels split for local: benchmarks.")
    ap.add_argument("--list_benchmarks", action="store_true", help="Print the registered benchmarks and exit.")
    ap.add_argument("--download_only", action="store_true", help="Fetch the benchmark data and exit.")

    ap.add_argument("--model", "--model_path", dest="model_path",
                    help="Local path or Hub id of a merged model, or of the base model when --lora_path is given. "
                         "Default: the released checkpoint chosen by --method / --size.")
    ap.add_argument("--method", choices=models.RELEASED_METHODS, default=models.DEFAULT_METHOD,
                    help="Which released checkpoint to use when --model is not given.")
    ap.add_argument("--size", default=models.DEFAULT_SIZE, help="Released checkpoint size (see `eklav models`).")
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--lora_rank", type=int, default=32, help="Must be >= the adapter's rank.")
    ap.add_argument("--model_tag", help="Label stored in every output row (default: the checkpoint's name).")
    ap.add_argument("--out_dir", help="Default: results/<task>/<model_tag>/<benchmark>.")
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=None, help="Default: 20000 (passage), 32768 (table).")
    ap.add_argument("--max_output_tokens", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)

    ap.add_argument("--num_questions", type=int, default=None, help="Seeded random sample of this many queries.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=None, help="Passage reranking: candidates judged per query (default 100).")
    ap.add_argument("--first_stage", choices=("bm25", "hybrid"), default=None,
                    help="Candidate pools for benchmarks that ship none (default: bm25 for passage, hybrid for table).")
    ap.add_argument("--skip_splade", action="store_true", help="Hybrid first stage without SPLADE.")
    ap.add_argument("--retrieval_device", default=None, help="Device for the hybrid first stage (default cuda:0).")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.list_benchmarks:
        for name in benchmarks.names():
            subsets = benchmarks.resolve(name).subsets
            print(f"{name}" + (f"  ({len(subsets)} subsets: {', '.join(subsets)})" if subsets else ""))
        print("local:<dir>   BEIR-layout directory\nhf:<repo>     Hub dataset with default/queries/corpus configs")
        return 0

    if not args.benchmark:
        ap.error("--benchmark is required")
    benchmark = benchmarks.resolve(args.benchmark, args.split)
    subsets = args.subsets or list(benchmark.subset_names())
    unknown = set(subsets) - set(benchmark.subset_names())
    if unknown:
        ap.error(f"Unknown subsets {sorted(unknown)} for {benchmark.name}; available: {list(benchmark.subset_names())}")

    if args.download_only:
        for subset in subsets:
            data = benchmark.loader(subset)
            print(f"[{benchmark.name}/{subset}] {len(data.queries)} queries, {len(data.corpus)} documents")
        return 0

    if not args.task:
        ap.error("--task is required")
    if args.lora_path and not args.model_path:
        ap.error("--lora_path needs --model (the base model the adapter was trained on)")
    if not args.model_path:
        try:
            ckpt = models.find_checkpoint(args.task, args.method, args.size)
        except ValueError as e:
            ap.error(str(e))
        args.model_path = ckpt.repo
        args.model_tag = args.model_tag or ckpt.repo.split("/")[-1]
    if not args.model_tag:
        ap.error("--model_tag is required with --model")
    args.out_dir = args.out_dir or os.path.join("results", args.task, args.model_tag, benchmark.name)
    os.makedirs(args.out_dir, exist_ok=True)

    model_args = ModelArgs(args.model_path, args.lora_path, args.lora_rank, args.num_gpus, args.max_model_len,
                           args.max_output_tokens, args.gpu_memory_utilization)
    evaluator = make_evaluator(args.task, model_args, top_k=args.top_k, first_stage=args.first_stage,
                               skip_splade=args.skip_splade or None, retrieval_device=args.retrieval_device)

    def prepare(subset: str):
        data = benchmark.loader(subset)
        return evaluator.prepare(benchmark.name, subset, data, benchmarks.select_queries(data, args.num_questions, args.seed))

    if evaluator.prepare_all_first:
        prepared = {s: prepare(s) for s in subsets}
        jobs = ((s, lambda s=s: prepared[s]) for s in subsets)
    else:
        jobs = ((s, lambda s=s: prepare(s)) for s in subsets)

    summaries: Dict[str, Dict] = {}
    for subset, get_prepared in jobs:
        start = time.time()
        summary = evaluator.run(get_prepared(), args.out_dir, args.model_tag)
        summaries[subset] = summary
        write_json(os.path.join(args.out_dir, f"{subset}_summary.json"), summary)
        print(f"[{subset}] nDCG@10={summary.get('ndcg@10', 0):.4f} ({time.time() - start:.0f}s)")

    overall = {"benchmark": benchmark.name, "model_tag": args.model_tag, "per_subset": summaries}
    if len(summaries) > 1:
        overall["average"] = average(summaries)
        print(f"average nDCG@10 = {overall['average']['ndcg@10']:.4f}")
    write_json(os.path.join(args.out_dir, "summary.json"), overall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
