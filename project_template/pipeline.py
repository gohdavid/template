"""Hydra entrypoint for standalone data and benchmark pipelines."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from project_template.pipeline_runtime import run_pipeline


@hydra.main(config_path="conf", config_name="pipeline_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)
