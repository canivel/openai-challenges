# Parameter Golf Challenge - Project Guide

## Overview
This repo contains our work for the OpenAI Parameter Golf challenge:
Train the best small language model fitting in 16MB, in 10 min on 8x H100 GPUs.

## Repository Structure
```
parameter-golf/          # Challenge repo (cloned from OpenAI)
research-framework/      # Our autonomous research framework
  src/                   # Python framework code
  agents/                # Agent prompt templates (in-framework)
  experiments/           # Experiment YAML configs
  results/               # Experiment results (TSV + JSON)
autoresearch/            # Karpathy's autoresearch (reference)
deer-flow/               # ByteDance's DeerFlow (reference)
.claude/agents/          # Claude Code agent definitions
```

## Key Metrics
- **BPB** (bits per byte): Primary metric. Lower is better.
- **Artifact size**: Code + compressed model <= 16,000,000 bytes
- **Training time**: <= 10 minutes on 8x H100 SXM
- **Current SOTA**: 1.1228 BPB

## Agent System
We have 5 specialized agents in `.claude/agents/`:
- **research-orchestrator**: Coordinates all research
- **architecture-agent**: Explores model architectures
- **hyperparam-agent**: Tunes training hyperparameters
- **quantization-agent**: Optimizes compression/quantization
- **eval-agent**: Validates results and prepares submissions
- **literature-agent**: Discovers new techniques

## Experiment Workflow
1. Orchestrator plans experiments based on current results
2. Each experiment runs in its own git worktree (parallel)
3. Agents modify `train_gpt.py` and run training
4. Results logged to `research-framework/results/`
5. Improvements kept (committed), failures reverted
6. Next iteration planned based on learnings

## Tool Preferences
- **ALWAYS use `uv`** for Python package management. Never use bare `pip`.
  - `uv pip install` instead of `pip install`
  - `uv run` instead of `python` for running scripts
  - `uv venv` for creating virtual environments

## Important Rules
- ALL model code must be in a single `train_gpt.py`
- No external downloads during evaluation
- New SOTA needs 3+ seed runs with p < 0.01
- Beat existing SOTA by >= 0.005 nats for a record
