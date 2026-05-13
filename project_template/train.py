"""Hydra entrypoint for training."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from project_template.train_runtime import run_training


@hydra.main(config_path="conf", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_training(cfg)
