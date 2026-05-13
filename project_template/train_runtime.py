from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from project_template.utils.config import (
    ensure_subdir,
    manifest_payload,
    run_dir,
    run_hash,
    validate_training_config,
    write_config_snapshot,
    write_manifest,
    write_run_metadata,
)


def run_training(cfg: DictConfig) -> None:
    """Minimal deterministic training loop placeholder.

    Replace this function with a real TrainingSession while preserving the
    output contract: hashed run directory, resolved config, metadata, metrics,
    manifest, and persisted state.
    """

    validate_training_config(cfg)
    run_hash_value, run_hash_full = run_hash(cfg)
    out_dir = run_dir(cfg, run_hash_value=run_hash_value)
    write_config_snapshot(out_dir, cfg)
    state_dir = ensure_subdir(out_dir, "state")
    resume = bool(OmegaConf.select(cfg, "train.resume", default=True))
    if not resume:
        _clear_training_state(state_dir)
    write_run_metadata(
        out_dir,
        cfg,
        entrypoint="train",
        run_hash_value=run_hash_value,
        run_hash_full=run_hash_full,
    )

    seed = int(cfg.seed)
    steps = int(cfg.train.num_steps)
    log_every = max(1, int(cfg.train.log_every))
    state_every = max(1, int(cfg.train.state_every))
    rng = np.random.default_rng(seed)

    latest_state = state_dir / "latest.npz"
    if resume and latest_state.exists():
        state = np.load(latest_state)
        start_step = int(state["step"])
        weight = float(state["weight"])
        if start_step >= steps:
            metrics = {
                "final_loss": float(state["loss"]),
                "final_weight": weight,
                "num_steps": start_step,
                "resumed": True,
            }
            _write_json(out_dir / "metrics.json", metrics)
            write_manifest(
                out_dir,
                manifest_payload(
                    entrypoint="train",
                    run_hash_value=run_hash_value,
                    artifacts=[
                        "config.yaml",
                        "run_metadata.json",
                        "metrics.json",
                        "state/latest.npz",
                    ],
                    status="already_complete",
                ),
            )
            print(f"training_run={out_dir} already_at_step={start_step}", flush=True)
            return
        start_step += 1
        print(f"resuming training_run={out_dir} from_step={start_step}", flush=True)
    else:
        start_step = 0
        weight = float(rng.normal())

    target = float(OmegaConf.select(cfg, "dataset.target", default=1.0))
    lr = float(OmegaConf.select(cfg, "optimizer.lr", default=0.01))
    history: list[dict[str, float | int]] = []
    log_steps = set(range(start_step, steps + 1, log_every))
    log_steps.add(steps)
    state_steps = set(range(start_step, steps + 1, state_every))
    state_steps.add(steps)
    log_actions = {scheduled_step: _log_step for scheduled_step in log_steps}
    state_actions = {scheduled_step: _write_state for scheduled_step in state_steps}

    final_loss = float((weight - target) ** 2)
    for step in range(start_step, steps + 1):
        loss = (weight - target) ** 2
        log_actions.get(step, _noop)(history, step, weight, loss)
        state_actions.get(step, _noop)(state_dir / f"step_{step:08d}.npz", step, weight, loss)
        weight -= lr * 2.0 * (weight - target)
        final_loss = float((weight - target) ** 2)

    metrics = {
        "final_loss": final_loss,
        "final_weight": float(weight),
        "num_steps": steps,
        "resumed": bool(resume and latest_state.exists()),
    }
    _write_json(out_dir / "metrics.json", metrics)
    np.savez_compressed(out_dir / "history.npz", rows=np.asarray(history, dtype=object))
    _write_state(state_dir / "latest.npz", steps, weight, final_loss)
    write_manifest(
        out_dir,
        manifest_payload(
            entrypoint="train",
            run_hash_value=run_hash_value,
            artifacts=[
                "config.yaml",
                "run_metadata.json",
                "metrics.json",
                "history.npz",
                "state/latest.npz",
            ],
            status="finished",
        ),
    )
    print(f"wrote training_run={out_dir}", flush=True)


def _log_step(
    history: list[dict[str, float | int]],
    step: int,
    weight: float,
    loss: float,
) -> None:
    row = {"step": step, "loss": float(loss), "weight": float(weight)}
    history.append(row)
    print(json.dumps(row, sort_keys=True), flush=True)


def _noop(*args, **kwargs) -> None:
    return None


def _write_state(path: Path, step: int, weight: float, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, step=np.asarray(step), weight=np.asarray(weight), loss=np.asarray(loss))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _clear_training_state(state_dir: Path) -> None:
    for path in state_dir.glob("*.npz"):
        path.unlink()
