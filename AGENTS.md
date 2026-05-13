# Template Agent Instructions

This file is for agents operating on this template repository itself. Read it
first when you are handed this template path.

Do not copy this file into projects created from the template. When starting or
retrofitting a real project, use these instructions to shape the project, then
write project-specific engineering and handoff guidance inside that project's
`.memory/` directory. The target project should not inherit this template's
root `AGENTS.md`, and it should not contain
`AGENT_ENGINEERING_PROMPT.md`.

## Read Order

Before editing this template, read:

1. `README.md`
2. `.memory/README.md`

The root `AGENTS.md` explains how to use the template. The `.memory/` directory
explains how a generated project should store its own engineering contract,
handoff state, research plan, and backlog.

## Template Contract

Preserve these defaults unless the user explicitly changes the template design:

- Python/Hydra-first. No Julia or language-agnostic abstractions.
- Root-level Python package, not `src/`.
- Pixi/uv execution through `./px`.
- Package-local Hydra configs.
- Explicit root configs:
  - `train_config.yaml`
  - `sample_config.yaml`
  - `eval_config.yaml`
  - `pipeline_config.yaml`
- Thin entrypoints:
  - `train.py` -> `train_runtime.py`
  - `sample.py` -> `sample_runtime.py`
  - `eval.py` -> `eval_runtime.py`
  - `pipeline.py` -> `pipeline_runtime.py`
- Use `name`, not `experiment_name`.
- Canonical artifacts under `artifacts/<name>/<hash>/`.
- Training state under `state/`, not top-level `checkpoints/`.
- Source-run-derived outputs attach under the source run:
  - `sample/<sample_name>/<sample_hash>/`
  - `eval/<eval_name>/<eval_hash>/`
  - `analysis/<analysis_name>/<analysis_hash>/`
- Stable hashes come from resolved configs with operational keys excluded.
- The `train/` config group is excluded from the model hash so training can
  resume or extend without changing model/data identity.
- Sampling, evaluation, and pipeline outputs use `on_existing: skip|overwrite`
  instead of hard failures when outputs already exist.
- Every run writes `config.yaml`, `run_metadata.json`, and `manifest.json`.
- Default model config stays generic unless the concrete project needs model
  substructure.
- Launcher names follow the embedded-style set:
  - `normal_h100`, `normal_h200`, `normal_l40s`
  - `preempt_h100`, `preempt_h200`, `preempt_l40s`

## Mode 1: Start A New Project From This Template

Use this workflow when the target is empty or should be created from this
template.

Required inputs:

```text
Template path: [TEMPLATE_PATH]
New project path: [TARGET_PROJECT_PATH]
Project name: [PROJECT_NAME]
Domain summary: [ONE_OR_TWO_SENTENCES_ABOUT_THE_PROJECT]
```

Workflow:

1. Copy or scaffold the template into `[TARGET_PROJECT_PATH]`.
2. Remove template-only root agent files from the target:
   - remove `AGENTS.md`
   - remove `AGENT_ENGINEERING_PROMPT.md` if it exists
3. Rename the Python package, console scripts, Pixi project name, pyproject
   metadata, README title, and package imports.
4. Replace placeholder runtime logic only where the project goal requires it.
5. Keep generic default config filenames until real alternatives exist.
6. Rewrite `.memory/README.md` for the concrete project. If repeated handoffs
   are expected, create the project memory files described there.
7. Keep generated artifacts, caches, `.venv`, `.pixi`, `.cache`, `outputs/`,
   `multirun/`, and `logs/` out of source control.
8. Run the smallest meaningful checks if dependencies are available.

The target project's durable agent context belongs in `.memory/`, not in this
template's root `AGENTS.md`.

Expected final response:

- Renamed files/packages and command names.
- Config roots and entrypoints.
- Artifact behavior.
- `.memory/` files created or updated.
- Checks run, or the exact next check command if checks were not run.

## Mode 2: Modify An Existing Project To Match This Template

Use this workflow when the target already has code, configs, scripts, or
artifacts that need to be brought into alignment.

Required inputs:

```text
Template path: [TEMPLATE_PATH]
Existing project path: [EXISTING_PROJECT_PATH]
Project name: [PROJECT_NAME]
```

Audit before editing:

- Current package layout and console scripts.
- Hydra config roots and config groups.
- Train/sample/eval/pipeline entrypoints.
- Artifact/checkpoint/output directory conventions.
- Hashing or reproducibility utilities.
- Pixi/uv setup, launcher configs, and SLURM scripts.
- Existing tests and smoke commands.
- Uncommitted user changes.

Refactor rules:

- Do not preserve backward compatibility unless explicitly requested.
- Do not add deprecated aliases, legacy config keys, or path shims.
- Do not keep duplicate old and new entrypoints.
- Preserve project-specific scientific logic; change structure, not meaning.
- Work with uncommitted user changes. Do not revert them.
- Do not copy this template's `AGENTS.md` into the target project.
- Add or update the target `.memory/` so future agents inherit
  project-specific state, not template history.

Expected final response:

- Migration summary mapping old structure to new structure.
- Files changed and important deletions.
- `.memory/` files created or updated.
- Any unresolved mismatches.
- Checks run, or the exact next check command if checks were not run.

## What To Put In `.memory/`

For a real project, store agent-facing engineering and handoff guidance in
`.memory/`, not in copied template root files. The template provides
`.memory/README.md` as the seed for:

- the project-specific engineering contract,
- current handoff state,
- durable architecture rules,
- research plan and motivation,
- backlog and optional hardening notes.

Small projects can keep only `.memory/README.md`. Larger or frequently handed
off projects should split it into the files described there.

## Prompt Templates

Use these prompts when asking another agent to apply this template. Keep the
warning about root agent files in the prompt.

### Create A New Project

```text
Use this template to create a new project:

Template path: /orcd/compute/edsun/001/davidgoh/repos/template
New project path: [ABSOLUTE_PATH_TO_NEW_REPO]
Project name: [PROJECT_NAME]
Domain summary: [ONE_OR_TWO_SENTENCES]

Read the template's AGENTS.md first. This is a template-only instruction file:
do not copy AGENTS.md or AGENT_ENGINEERING_PROMPT.md into the target
repo.

Create the target project with the template's Python/Hydra research-project
structure:
- root-level Python package, not src/
- Pixi/uv execution through ./px
- package-local Hydra configs
- explicit train_config.yaml, sample_config.yaml, eval_config.yaml,
  pipeline_config.yaml
- thin entrypoints: train.py -> train_runtime.py, sample.py ->
  sample_runtime.py, eval.py -> eval_runtime.py, pipeline.py ->
  pipeline_runtime.py
- use name, not experiment_name
- artifacts under artifacts/<name>/<hash>/
- training state under state/
- source-run-derived sample/eval/analysis outputs attached under the source run
- stable hashes from resolved configs with operational keys excluded
- train config excluded from model hash so training can resume/extend
- skip|overwrite behavior for existing sample/eval/pipeline outputs

Write the target repo's durable agent guidance in .memory/, not in copied
template root files. Rewrite .memory/README.md for this project and create
additional memory files if repeated agent handoffs are expected.

After creating the project, run the smallest meaningful checks available.

Final response should include:
- renamed files/packages and command names
- config roots and entrypoints
- artifact behavior
- .memory files created or updated
- checks run, or exact next check command if checks could not be run
```

### Retrofit An Existing Project

```text
Use this template to retrofit the current repo:

Template path: /orcd/compute/edsun/001/davidgoh/repos/template
Existing project path: [ABSOLUTE_PATH_TO_TARGET_REPO]
Project name: [PROJECT_NAME]

Read the template's AGENTS.md first. This is a template-only instruction file:
do not copy AGENTS.md or AGENT_ENGINEERING_PROMPT.md into the target
repo.

Refactor the target repo to match the template's Python/Hydra research-project
structure:
- root-level Python package, not src/
- Pixi/uv execution through ./px
- package-local Hydra configs
- explicit train_config.yaml, sample_config.yaml, eval_config.yaml,
  pipeline_config.yaml
- thin entrypoints: train.py -> train_runtime.py, sample.py ->
  sample_runtime.py, eval.py -> eval_runtime.py, pipeline.py ->
  pipeline_runtime.py
- use name, not experiment_name
- artifacts under artifacts/<name>/<hash>/
- training state under state/
- source-run-derived sample/eval/analysis outputs attached under the source run
- stable hashes from resolved configs with operational keys excluded
- train config excluded from model hash so training can resume/extend
- skip|overwrite behavior for existing sample/eval/pipeline outputs

Preserve the target repo's scientific logic. Change structure and
reproducibility conventions, not scientific meaning. Do not keep backward-
compatibility shims or duplicate old/new entrypoints unless explicitly asked.

Update the target repo's .memory/ directory with project-specific engineering
instructions, handoff notes, architecture contract, research context, and
backlog as needed. Durable agent guidance for the target project belongs in
.memory/, not in copied template root files.

Before editing, audit:
- current package layout and console scripts
- Hydra config roots and config groups
- train/sample/eval/pipeline entrypoints
- artifact/checkpoint/output conventions
- hashing/reproducibility utilities
- Pixi/uv setup, launcher configs, and SLURM scripts
- existing tests and smoke commands
- uncommitted user changes

After editing, run the smallest meaningful checks available.

Final response should include:
- migration summary mapping old structure to new structure
- files changed and important deletions
- .memory files created or updated
- unresolved mismatches
- checks run, or exact next check command if checks could not be run
```
