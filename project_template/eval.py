"""Hydra entrypoint for source-run evaluation."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from project_template.eval_runtime import run_evaluation


@hydra.main(config_path="conf", config_name="eval_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)
