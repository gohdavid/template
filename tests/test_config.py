from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from project_template.eval_runtime import run_evaluation
from project_template.pipeline_runtime import run_pipeline
from project_template.sample_runtime import run_sampling
from project_template.train_runtime import run_training
from project_template.utils.config import (
    HASH_EXCLUDE_PROFILES,
    attached_run_dir,
    config_hash,
    require_source_run_dir,
    run_dir,
    run_hash,
    write_config_snapshot,
)


def test_hash_ignores_output() -> None:
    base = OmegaConf.create(
        {
            "seed": 0,
            "model": {"hidden_dim": 16},
            "output": {"root": "runs_a"},
            "hash": {"profile": "pipeline", "extra_exclude": []},
        }
    )
    changed = OmegaConf.merge(base, {"output": {"root": "runs_b"}})
    assert config_hash(base, exclude=HASH_EXCLUDE_PROFILES["pipeline"]) == config_hash(
        changed,
        exclude=HASH_EXCLUDE_PROFILES["pipeline"],
    )


def test_run_dir_uses_hash(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "demo",
            "seed": 0,
            "output": {"root": str(tmp_path)},
            "hash": {"profile": "pipeline", "extra_exclude": [], "n_chars": 8},
        }
    )
    path = run_dir(
        cfg,
        run_hash_value=config_hash(
            cfg,
            exclude=HASH_EXCLUDE_PROFILES["pipeline"],
            n_chars=8,
        ),
    )
    assert path.parent.name == "demo"
    assert len(path.name) == 8
    assert path.exists()


def test_attached_run_dir_stays_under_parent(tmp_path) -> None:
    cfg = OmegaConf.create({"output": {"name": "loss_curves"}})
    parent = tmp_path / "train_exp" / "abc123"
    path = attached_run_dir(parent, cfg, group="analysis", run_hash_value="def456")
    assert path == parent / "analysis" / "loss_curves" / "def456"
    assert path.exists()


def test_model_hash_ignores_train_policy() -> None:
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "dataset": {"name": "default"},
            "model": {"hidden_dim": 16},
            "train": {"batch_size": 8, "num_steps": 10},
            "hash": {"profile": "model", "n_chars": 12, "extra_exclude": []},
        }
    )
    extended = OmegaConf.merge(cfg, {"train": {"num_steps": 1000}})
    changed_batch = OmegaConf.merge(cfg, {"train": {"batch_size": 64}})

    assert run_hash(cfg)[0] == run_hash(extended)[0]
    assert run_hash(cfg)[0] == run_hash(changed_batch)[0]


def test_existing_policy_does_not_change_hash() -> None:
    sample_cfg = OmegaConf.create(
        {
            "name": "default",
            "seed": 0,
            "source": {"run_dir": "artifacts/training/source"},
            "sample": {"n_samples": 4, "on_existing": "skip"},
            "hash": {"profile": "sample", "n_chars": 12, "extra_exclude": []},
        }
    )
    eval_cfg = OmegaConf.create(
        {
            "name": "default",
            "seed": 0,
            "source": {"run_dir": "artifacts/training/source"},
            "eval": {"n_bootstrap": 4, "on_existing": "skip"},
            "hash": {"profile": "eval", "n_chars": 12, "extra_exclude": []},
        }
    )
    pipeline_cfg = OmegaConf.create(
        {
            "name": "pipeline",
            "seed": 0,
            "pipeline": {"n_samples": 4, "on_existing": "skip"},
            "hash": {"profile": "pipeline", "n_chars": 12, "extra_exclude": []},
        }
    )

    assert run_hash(sample_cfg)[0] == run_hash(
        OmegaConf.merge(sample_cfg, {"sample": {"on_existing": "overwrite"}})
    )[0]
    assert run_hash(eval_cfg)[0] == run_hash(
        OmegaConf.merge(eval_cfg, {"eval": {"on_existing": "overwrite"}})
    )[0]
    assert run_hash(pipeline_cfg)[0] == run_hash(
        OmegaConf.merge(pipeline_cfg, {"pipeline": {"on_existing": "overwrite"}})
    )[0]


def test_require_source_run_dir_requires_config_snapshot(tmp_path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        require_source_run_dir(missing)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError, match="missing config.yaml"):
        require_source_run_dir(incomplete)

    (incomplete / "config.yaml").write_text("name: incomplete\n", encoding="utf-8")
    assert require_source_run_dir(incomplete) == incomplete


def test_config_snapshot_overwrites_existing_snapshot(tmp_path) -> None:
    cfg = OmegaConf.create({"name": "demo", "seed": 0})
    changed = OmegaConf.merge(cfg, {"seed": 1})
    path = tmp_path / "demo" / "abc123"

    write_config_snapshot(path, cfg)
    write_config_snapshot(path, changed)

    assert "seed: 1" in (path / "config.yaml").read_text(encoding="utf-8")


def test_sampling_accepts_incomplete_source_with_config_snapshot(tmp_path) -> None:
    source_dir = tmp_path / "training" / "abc123"
    source_dir.mkdir(parents=True)
    (source_dir / "config.yaml").write_text("name: training\n", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "name": "default",
            "seed": 0,
            "source": {"run_dir": str(source_dir)},
            "output": {"name": "default"},
            "hash": {"profile": "sample", "n_chars": 12, "extra_exclude": []},
            "sample": {"n_samples": 4, "loc": 0.0, "scale": 1.0},
        }
    )
    sample_hash, _ = run_hash(cfg)

    run_sampling(cfg)

    assert (source_dir / "sample" / "default" / sample_hash / "samples.npz").exists()


def _read_manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _assert_manifest_shape(
    manifest: dict,
    *,
    entrypoint: str,
    source_run_dir: str | None,
    status: str,
) -> None:
    assert manifest["entrypoint"] == entrypoint
    assert manifest["source_run_dir"] == source_run_dir
    assert manifest["status"] == status
    assert manifest["artifacts"]
    assert manifest["run_hash"]


def test_entrypoint_runtime_smoke_contracts(tmp_path) -> None:
    with initialize_config_module(
        config_module="project_template.conf",
        version_base=None,
    ):
        train_cfg = compose(
            config_name="train_config",
            overrides=[f"output.root={tmp_path.as_posix()}", "train=smoke"],
        )
        train_hash, _ = run_hash(train_cfg)
        train_dir = Path(tmp_path) / "training" / train_hash
        run_training(train_cfg)

        assert (train_dir / "config.yaml").exists()
        assert (train_dir / "run_metadata.json").exists()
        assert (train_dir / "metrics.json").exists()
        assert (train_dir / "state" / "latest.npz").exists()
        assert (train_dir / "manifest.json").exists()
        assert not (train_dir / "complete.txt").exists()
        _assert_manifest_shape(
            _read_manifest(train_dir),
            entrypoint="train",
            source_run_dir=None,
            status="finished",
        )

        extended_train_cfg = compose(
            config_name="train_config",
            overrides=[
                f"output.root={tmp_path.as_posix()}",
                "train=smoke",
                "train.num_steps=12",
            ],
        )
        extended_train_hash, _ = run_hash(extended_train_cfg)
        assert extended_train_hash == train_hash
        run_training(extended_train_cfg)
        latest_state = np.load(train_dir / "state" / "latest.npz")
        assert int(latest_state["step"]) == 12

        run_training(extended_train_cfg)
        latest_state = np.load(train_dir / "state" / "latest.npz")
        assert int(latest_state["step"]) == 12
        _assert_manifest_shape(
            _read_manifest(train_dir),
            entrypoint="train",
            source_run_dir=None,
            status="already_complete",
        )

        restart_train_cfg = compose(
            config_name="train_config",
            overrides=[
                f"output.root={tmp_path.as_posix()}",
                "train=smoke",
                "train.num_steps=3",
                "train.resume=false",
            ],
        )
        restart_train_hash, _ = run_hash(restart_train_cfg)
        assert restart_train_hash == train_hash
        run_training(restart_train_cfg)
        latest_state = np.load(train_dir / "state" / "latest.npz")
        assert int(latest_state["step"]) == 3
        _assert_manifest_shape(
            _read_manifest(train_dir),
            entrypoint="train",
            source_run_dir=None,
            status="finished",
        )

        sample_cfg = compose(
            config_name="sample_config",
            overrides=[f"source.run_dir={train_dir.as_posix()}"],
        )
        sample_hash, _ = run_hash(sample_cfg)
        sample_dir = train_dir / "sample" / "default" / sample_hash
        run_sampling(sample_cfg)

        assert (sample_dir / "config.yaml").exists()
        assert (sample_dir / "samples.npz").exists()
        assert (sample_dir / "metrics.json").exists()
        assert (sample_dir / "manifest.json").exists()
        assert not (sample_dir / "complete.txt").exists()
        _assert_manifest_shape(
            _read_manifest(sample_dir),
            entrypoint="sample",
            source_run_dir=str(train_dir),
            status="finished",
        )

        run_sampling(sample_cfg)
        _assert_manifest_shape(
            _read_manifest(sample_dir),
            entrypoint="sample",
            source_run_dir=str(train_dir),
            status="skipped_existing",
        )

        sample_overwrite_cfg = compose(
            config_name="sample_config",
            overrides=[
                f"source.run_dir={train_dir.as_posix()}",
                "sample.on_existing=overwrite",
            ],
        )
        assert run_hash(sample_overwrite_cfg)[0] == sample_hash
        run_sampling(sample_overwrite_cfg)
        _assert_manifest_shape(
            _read_manifest(sample_dir),
            entrypoint="sample",
            source_run_dir=str(train_dir),
            status="finished",
        )

        eval_cfg = compose(
            config_name="eval_config",
            overrides=[f"source.run_dir={sample_dir.as_posix()}"],
        )
        eval_hash, _ = run_hash(eval_cfg)
        eval_dir = sample_dir / "eval" / "default" / eval_hash
        run_evaluation(eval_cfg)

        assert (eval_dir / "config.yaml").exists()
        assert (eval_dir / "scores.npz").exists()
        assert (eval_dir / "metrics.json").exists()
        assert (eval_dir / "manifest.json").exists()
        assert not (eval_dir / "complete.txt").exists()
        _assert_manifest_shape(
            _read_manifest(eval_dir),
            entrypoint="eval",
            source_run_dir=str(sample_dir),
            status="finished",
        )

        run_evaluation(eval_cfg)
        _assert_manifest_shape(
            _read_manifest(eval_dir),
            entrypoint="eval",
            source_run_dir=str(sample_dir),
            status="skipped_existing",
        )

        eval_overwrite_cfg = compose(
            config_name="eval_config",
            overrides=[
                f"source.run_dir={sample_dir.as_posix()}",
                "eval.on_existing=overwrite",
            ],
        )
        assert run_hash(eval_overwrite_cfg)[0] == eval_hash
        run_evaluation(eval_overwrite_cfg)
        _assert_manifest_shape(
            _read_manifest(eval_dir),
            entrypoint="eval",
            source_run_dir=str(sample_dir),
            status="finished",
        )

        pipeline_cfg = compose(
            config_name="pipeline_config",
            overrides=[f"output.root={tmp_path.as_posix()}"],
        )
        pipeline_hash, _ = run_hash(pipeline_cfg)
        pipeline_dir = Path(tmp_path) / "pipeline" / pipeline_hash
        run_pipeline(pipeline_cfg)

        assert (pipeline_dir / "config.yaml").exists()
        assert (pipeline_dir / "values.npz").exists()
        assert (pipeline_dir / "metrics.json").exists()
        assert (pipeline_dir / "manifest.json").exists()
        assert not (pipeline_dir / "complete.txt").exists()
        _assert_manifest_shape(
            _read_manifest(pipeline_dir),
            entrypoint="pipeline",
            source_run_dir=None,
            status="finished",
        )

        run_pipeline(pipeline_cfg)
        _assert_manifest_shape(
            _read_manifest(pipeline_dir),
            entrypoint="pipeline",
            source_run_dir=None,
            status="skipped_existing",
        )

        pipeline_overwrite_cfg = compose(
            config_name="pipeline_config",
            overrides=[
                f"output.root={tmp_path.as_posix()}",
                "pipeline.on_existing=overwrite",
            ],
        )
        assert run_hash(pipeline_overwrite_cfg)[0] == pipeline_hash
        run_pipeline(pipeline_overwrite_cfg)
        _assert_manifest_shape(
            _read_manifest(pipeline_dir),
            entrypoint="pipeline",
            source_run_dir=None,
            status="finished",
        )
