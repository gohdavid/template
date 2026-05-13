from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from project_template.utils.config import (
    attached_run_dir,
    manifest_payload,
    run_hash,
    should_skip_existing_outputs,
    validate_evaluation_config,
    write_config_snapshot,
    write_manifest,
    write_run_metadata,
)


def run_evaluation(cfg: DictConfig) -> None:
    """Evaluate a source run or a derived sample run."""

    validate_evaluation_config(cfg)
    run_hash_value, run_hash_full = run_hash(cfg)
    source_run_dir = Path(str(cfg.source.run_dir))
    out_dir = attached_run_dir(
        source_run_dir,
        cfg,
        group="eval",
        run_hash_value=run_hash_value,
    )
    write_config_snapshot(out_dir, cfg)
    write_run_metadata(
        out_dir,
        cfg,
        entrypoint="eval",
        run_hash_value=run_hash_value,
        run_hash_full=run_hash_full,
    )

    existing_policy = str(OmegaConf.select(cfg, "eval.on_existing", default="skip"))
    scores_path = out_dir / "scores.npz"
    metrics_path = out_dir / "metrics.json"
    if should_skip_existing_outputs(
        paths=(scores_path, metrics_path),
        on_existing=existing_policy,
    ):
        write_manifest(
            out_dir,
            manifest_payload(
                entrypoint="eval",
                run_hash_value=run_hash_value,
                source_run_dir=source_run_dir,
                artifacts=[
                    "config.yaml",
                    "run_metadata.json",
                    "metrics.json",
                    "scores.npz",
                ],
                status="skipped_existing",
            ),
        )
        print(OmegaConf.to_yaml({"output_dir": str(out_dir), "status": "skipped_existing"}))
        return

    rng = np.random.default_rng(int(cfg.seed))
    n = int(OmegaConf.select(cfg, "eval.n_bootstrap", default=32))
    scores = rng.uniform(size=n)
    metrics = {
        "source_exists": True,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
    }

    np.savez_compressed(scores_path, scores=scores)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        out_dir,
        manifest_payload(
            entrypoint="eval",
            run_hash_value=run_hash_value,
            source_run_dir=source_run_dir,
            artifacts=[
                "config.yaml",
                "run_metadata.json",
                "metrics.json",
                "scores.npz",
            ],
            status="finished",
        ),
    )
    print(OmegaConf.to_yaml({"output_dir": str(out_dir), "metrics": metrics}))
