"""Hydra entrypoint for source-run sampling."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from project_template.sample_runtime import run_sampling


@hydra.main(config_path="conf", config_name="sample_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_sampling(cfg)
