from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

RUNTIME_HASH_EXCLUDE = ("name", "output", "hydra", "hash", "wandb", "slurm")

HASH_EXCLUDE_PROFILES = {
    "model": RUNTIME_HASH_EXCLUDE + ("train", "sample", "eval", "pipeline", "source"),
    "sample": RUNTIME_HASH_EXCLUDE + ("source", "sample.on_existing"),
    "eval": RUNTIME_HASH_EXCLUDE + ("source", "eval.on_existing"),
    "pipeline": RUNTIME_HASH_EXCLUDE + ("pipeline.on_existing",),
}


def resolved_config(cfg: DictConfig) -> dict:
    return dict(OmegaConf.to_container(cfg, resolve=True))


def config_digest(
    cfg: DictConfig,
    *,
    exclude: Iterable[str] = RUNTIME_HASH_EXCLUDE,
) -> str:
    payload = resolved_config(cfg)
    for key in exclude:
        drop_config_key(payload, str(key))
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_hash(
    cfg: DictConfig,
    *,
    exclude: Iterable[str] = RUNTIME_HASH_EXCLUDE,
    n_chars: int = 12,
) -> str:
    return config_digest(cfg, exclude=exclude)[:n_chars]


def hash_exclude_from_cfg(cfg: DictConfig) -> tuple[str, ...]:
    profile = str(OmegaConf.select(cfg, "hash.profile", default="pipeline"))
    if profile not in HASH_EXCLUDE_PROFILES:
        options = ", ".join(sorted(HASH_EXCLUDE_PROFILES))
        raise ValueError(f"unknown hash.profile={profile!r}; expected one of: {options}")
    base = HASH_EXCLUDE_PROFILES[profile]
    extra = OmegaConf.select(cfg, "hash.extra_exclude", default=())
    return tuple(dict.fromkeys((*base, *(str(item) for item in extra))))


def hash_chars_from_cfg(cfg: DictConfig) -> int:
    return int(OmegaConf.select(cfg, "hash.n_chars", default=12))


def run_hash(cfg: DictConfig) -> tuple[str, str]:
    exclude = hash_exclude_from_cfg(cfg)
    digest = config_digest(cfg, exclude=exclude)
    return digest[: hash_chars_from_cfg(cfg)], digest


def run_dir(cfg: DictConfig, *, run_hash_value: str) -> Path:
    root = Path(str(OmegaConf.select(cfg, "output.root", default="artifacts")))
    name = str(
        OmegaConf.select(
            cfg,
            "output.name",
            default=OmegaConf.select(cfg, "name", default="experiment"),
        )
    )
    path = root / name / run_hash_value
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_subdir(path: Path, name: str) -> Path:
    path = path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def attached_run_dir(
    parent_run_dir: Path,
    cfg: DictConfig,
    *,
    group: str,
    run_hash_value: str,
) -> Path:
    name = str(
        OmegaConf.select(
            cfg,
            "output.name",
            default=OmegaConf.select(cfg, "name", default="default"),
        )
    )
    path = parent_run_dir / group / name / run_hash_value
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_common_config(cfg: DictConfig, *, entrypoint: str) -> None:
    if OmegaConf.select(cfg, "name", default=None) in (None, ""):
        raise ValueError(f"{entrypoint} config requires non-empty name")
    if int(OmegaConf.select(cfg, "hash.n_chars", default=12)) < 8:
        raise ValueError(f"{entrypoint} config hash.n_chars must be at least 8")


def validate_training_config(cfg: DictConfig) -> None:
    validate_common_config(cfg, entrypoint="train")
    _require_positive_int(cfg, "train.num_steps")
    _require_positive_int(cfg, "train.batch_size")
    _require_positive_int(cfg, "train.log_every")
    _require_positive_int(cfg, "train.state_every")


def validate_sampling_config(cfg: DictConfig) -> None:
    validate_common_config(cfg, entrypoint="sample")
    _require_positive_int(cfg, "sample.n_samples")
    require_source_run_dir(_source_run_dir(cfg))


def validate_evaluation_config(cfg: DictConfig) -> None:
    validate_common_config(cfg, entrypoint="eval")
    _require_positive_int(cfg, "eval.n_bootstrap")
    require_source_run_dir(_source_run_dir(cfg))


def validate_pipeline_config(cfg: DictConfig) -> None:
    validate_common_config(cfg, entrypoint="pipeline")
    _require_positive_int(cfg, "pipeline.n_samples")


def _require_positive_int(cfg: DictConfig, key: str) -> None:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        raise ValueError(f"config requires {key}")
    if int(value) <= 0:
        raise ValueError(f"config {key} must be positive")


def _source_run_dir(cfg: DictConfig) -> Path:
    value = OmegaConf.select(cfg, "source.run_dir", default=None)
    if value in (None, ""):
        raise ValueError("config requires source.run_dir")
    return Path(str(value))


def require_source_run_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"source.run_dir does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"source.run_dir is not a directory: {path}")
    if not (path / "config.yaml").exists():
        raise FileNotFoundError(f"source.run_dir is missing config.yaml: {path}")
    return path


def should_skip_existing_outputs(
    *,
    paths: Iterable[Path],
    on_existing: str,
) -> bool:
    policy = str(on_existing)
    if policy not in {"skip", "overwrite"}:
        raise ValueError("on_existing must be 'skip' or 'overwrite'")
    return policy == "skip" and all(path.exists() for path in paths)


def drop_config_key(payload: dict, key: str) -> None:
    parts = key.split(".")
    current = payload
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            return
        current = next_value
    current.pop(parts[-1], None)


def write_config_snapshot(path: Path, cfg: DictConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def write_manifest(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def manifest_payload(
    *,
    entrypoint: str,
    run_hash_value: str,
    artifacts: Iterable[str],
    source_run_dir: Path | None = None,
    status: str = "finished",
) -> dict:
    return {
        "entrypoint": entrypoint,
        "run_hash": run_hash_value,
        "source_run_dir": None if source_run_dir is None else str(source_run_dir),
        "status": status,
        "artifacts": list(artifacts),
    }


def write_run_metadata(
    path: Path,
    cfg: DictConfig,
    *,
    entrypoint: str,
    run_hash_value: str,
    run_hash_full: str,
) -> None:
    metadata = {
        "entrypoint": entrypoint,
        "run_hash": run_hash_value,
        "run_hash_full": run_hash_full,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    (path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None
