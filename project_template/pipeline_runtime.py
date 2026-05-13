from __future__ import annotations

import json

import numpy as np
from omegaconf import DictConfig, OmegaConf

from project_template.utils.config import (
    manifest_payload,
    run_dir,
    run_hash,
    should_skip_existing_outputs,
    validate_pipeline_config,
    write_config_snapshot,
    write_manifest,
    write_run_metadata,
)


def run_pipeline(cfg: DictConfig) -> None:
    """Minimal standalone pipeline placeholder.

    This mirrors the training output contract: one hashed run directory,
    resolved config, metadata, manifest, metrics, and artifacts.
    """

    validate_pipeline_config(cfg)
    run_hash_value, run_hash_full = run_hash(cfg)
    out_dir = run_dir(cfg, run_hash_value=run_hash_value)
    write_config_snapshot(out_dir, cfg)
    write_run_metadata(
        out_dir,
        cfg,
        entrypoint="pipeline",
        run_hash_value=run_hash_value,
        run_hash_full=run_hash_full,
    )

    values_path = out_dir / "values.npz"
    metrics_path = out_dir / "metrics.json"
    if should_skip_existing_outputs(
        paths=(values_path, metrics_path),
        on_existing=str(OmegaConf.select(cfg, "pipeline.on_existing", default="skip")),
    ):
        write_manifest(
            out_dir,
            manifest_payload(
                entrypoint="pipeline",
                run_hash_value=run_hash_value,
                artifacts=[
                    "config.yaml",
                    "run_metadata.json",
                    "metrics.json",
                    "values.npz",
                ],
                status="skipped_existing",
            ),
        )
        print(OmegaConf.to_yaml({"output_dir": str(out_dir), "status": "skipped_existing"}))
        return

    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)
    n = int(OmegaConf.select(cfg, "pipeline.n_samples", default=128))
    loc = float(OmegaConf.select(cfg, "pipeline.loc", default=0.0))
    scale = float(OmegaConf.select(cfg, "pipeline.scale", default=1.0))
    values = rng.normal(loc=loc, scale=scale, size=n)

    metrics = {
        "n_samples": n,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    np.savez_compressed(values_path, values=values)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        out_dir,
        manifest_payload(
            entrypoint="pipeline",
            run_hash_value=run_hash_value,
            artifacts=[
                "config.yaml",
                "run_metadata.json",
                "metrics.json",
                "values.npz",
            ],
        ),
    )
    print(OmegaConf.to_yaml({"output_dir": str(out_dir), "metrics": metrics}))
