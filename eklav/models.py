"""
The released checkpoints on the Hugging Face Hub.

    eklav models                                     # list them
    eklav download --task passage_reranking --method eklav --size 8b

`eklav eval` uses the same registry: with no --model it evaluates the released
checkpoint chosen by --task / --method / --size, which vLLM downloads on first use.
"""

from typing import Dict, List, NamedTuple, Optional

from eklav.methods import EKLAV, NAIVE_SFT, STD_SFT

HF_USER = "AdarshSingh7647"
DEFAULT_METHOD = EKLAV
DEFAULT_SIZE = "8b"


class Checkpoint(NamedTuple):
    task: str
    method: str
    size: str
    repo: str       # Hub repo id
    base: str       # base model the checkpoint was trained from

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}"


def _entries(task: str, method: str, base_models: Dict[str, str], names: Dict[str, str]) -> List[Checkpoint]:
    return [Checkpoint(task, method, size, f"{HF_USER}/{name}", base_models[size]) for size, name in names.items()]


_PASSAGE_BASES = {"0.6b": "Qwen/Qwen3-0.6B", "4b": "Qwen/Qwen3-4B", "8b": "Qwen/Qwen3-8B",
                  "9b": "zai-org/GLM-Z1-9B-0414", "14b": "Qwen/Qwen3-14B"}

CHECKPOINTS: List[Checkpoint] = (
    _entries("passage_reranking", EKLAV, _PASSAGE_BASES,
             {size: f"Eklav-Passage-{size.upper()}" for size in ("0.6b", "4b", "8b", "9b", "14b")})
    + _entries("passage_reranking", STD_SFT, _PASSAGE_BASES,
               {size: f"Std-SFT-Passage-{size.upper()}" for size in ("0.6b", "4b", "8b", "9b", "14b")})
    + _entries("table_reranking", EKLAV, {"8b": "Qwen/Qwen3-8B"}, {"8b": "Eklav-Table-8B"})
    + _entries("table_reranking", STD_SFT, {"8b": "Qwen/Qwen3-8B"}, {"8b": "Std-SFT-Table-8B"})
    + _entries("table_reranking", NAIVE_SFT, {"8b": "Qwen/Qwen3-8B"}, {"8b": "Naive-SFT-Table-8B"})
)
RELEASED_METHODS = tuple(dict.fromkeys(c.method for c in CHECKPOINTS))


def find_checkpoint(task: str, method: str, size: str) -> Checkpoint:
    for ckpt in CHECKPOINTS:
        if (ckpt.task, ckpt.method, ckpt.size) == (task, method, size.lower()):
            return ckpt
    available = sorted(c.size for c in CHECKPOINTS if (c.task, c.method) == (task, method))
    raise ValueError(f"No released {method} checkpoint for {task} at size {size!r}; "
                     f"available sizes: {', '.join(available) or 'none'}. See `eklav models`.")


def download(ckpt: Checkpoint, local_dir: Optional[str] = None) -> str:
    """Fetch a released checkpoint (a no-op when it is already cached); returns its local path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(ckpt.repo, local_dir=local_dir)


def format_table() -> str:
    rows = [("task", "method", "size", "base model", "checkpoint")]
    rows += [(c.task, c.method, c.size, c.base, c.repo) for c in CHECKPOINTS]
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows)

