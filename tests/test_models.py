import pytest

from eklav import cli, models
from eklav.eval import run


def test_every_checkpoint_resolves_and_is_unique():
    repos = [c.repo for c in models.CHECKPOINTS]
    assert len(repos) == len(set(repos))
    for c in models.CHECKPOINTS:
        assert models.find_checkpoint(c.task, c.method, c.size) == c
        assert c.url == f"https://huggingface.co/{c.repo}"


def test_unknown_checkpoint_lists_available_sizes():
    with pytest.raises(ValueError, match="available sizes: .*8b"):
        models.find_checkpoint("passage_reranking", "eklav", "3b")


def test_models_command_lists_checkpoints(capsys):
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "AdarshSingh7647/Eklav-8B-Reranker" in out and "TabRankStandardSFT" in out


def test_eval_defaults_to_the_released_checkpoint(monkeypatch, tmp_path):
    seen = {}

    class Stop(Exception):
        pass

    def fake_make_evaluator(task, model_args, **options):
        seen.update(task=task, model=model_args.model_path)
        raise Stop

    monkeypatch.setattr(run, "make_evaluator", fake_make_evaluator)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Stop):
        run.main(["--task", "passage_reranking", "--benchmark", "bright", "--method", "std-sft", "--size", "4b"])
    assert seen == {"task": "passage_reranking", "model": "AdarshSingh7647/Eklav-4B-Reranker-CotGen"}
    assert (tmp_path / "results/passage_reranking/Eklav-4B-Reranker-CotGen/bright").is_dir()


def test_eval_rejects_adapter_without_base_model():
    with pytest.raises(SystemExit):
        run.main(["--task", "passage_reranking", "--benchmark", "bright", "--lora_path", "adapter"])
