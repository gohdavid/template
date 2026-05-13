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
    validate_sampling_config,
    write_config_snapshot,
    write_manifest,
    write_run_metadata,
)


def run_sampling(cfg: DictConfig) -> None:
    """Create derived samples attached to a source run directory."""

    validate_sampling_config(cfg)
    run_hash_value, run_hash_full = run_hash(cfg)
    source_run_dir = Path(str(cfg.source.run_dir))
    out_dir = attached_run_dir(
        source_run_dir,
        cfg,
        group="sample",
        run_hash_value=run_hash_value,
    )
    write_config_snapshot(out_dir, cfg)
    write_run_metadata(
        out_dir,
        cfg,
        entrypoint="sample",
        run_hash_value=run_hash_value,
        run_hash_full=run_hash_full,
    )

    samples_path = out_dir / "samples.npz"
    metrics_path = out_dir / "metrics.json"
    if should_skip_existing_outputs(
        paths=(samples_path, metrics_path),
        on_existing=str(OmegaConf.select(cfg, "sample.on_existing", default="skip")),
    ):
        write_manifest(
            out_dir,
            manifest_payload(
                entrypoint="sample",
                run_hash_value=run_hash_value,
                source_run_dir=source_run_dir,
                artifacts=[
                    "config.yaml",
                    "run_metadata.json",
                    "metrics.json",
                    "samples.npz",
                ],
                status="skipped_existing",
            ),
        )
        print(OmegaConf.to_yaml({"output_dir": str(out_dir), "status": "skipped_existing"}))
        return

    rng = np.random.default_rng(int(cfg.seed))
    n = int(OmegaConf.select(cfg, "sample.n_samples", default=128))
    loc = float(OmegaConf.select(cfg, "sample.loc", default=0.0))
    scale = float(OmegaConf.select(cfg, "sample.scale", default=1.0))
    values = rng.normal(loc=loc, scale=scale, size=n)
    metrics = {"n_samples": n, "sample_mean": float(np.mean(values))}

    np.savez_compressed(samples_path, values=values)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        out_dir,
        manifest_payload(
            entrypoint="sample",
            run_hash_value=run_hash_value,
            source_run_dir=source_run_dir,
            artifacts=[
                "config.yaml",
                "run_metadata.json",
                "metrics.json",
                "samples.npz",
            ],
        ),
    )
    print(OmegaConf.to_yaml({"output_dir": str(out_dir), "metrics": metrics}))
