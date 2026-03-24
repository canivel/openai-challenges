# Golf Research Framework

An autonomous ML research framework for the Parameter Golf challenge, combining the best patterns from [autoresearch](https://github.com/karpathy/autoresearch) and [DeerFlow](https://github.com/bytedance/deer-flow).

## Design Philosophy

### From Autoresearch: Simplicity & Autonomy
- **Single-file mutation**: Each experiment modifies one `train_gpt.py` and measures the result
- **Metric-driven decisions**: If BPB improved → keep (commit). If not → revert (git reset)
- **Git-based tracking**: Every experiment is a commit, every failure is visible in history
- **Autonomous loops**: Agents run continuously without human intervention

### From DeerFlow: Scale & Structure
- **Parallel agents**: Multiple specialized agents work simultaneously in isolated git worktrees
- **Structured decomposition**: Break the optimization problem into architecture / hyperparameters / quantization
- **Memory persistence**: Track discoveries and failed approaches across sessions
- **Middleware pipeline**: Orchestrator coordinates agents, adapts strategy based on results

### Our Innovation: Worktree-Based Parallelism
Each agent gets its own git worktree — a full copy of the repository on its own branch. This means:
- Architecture agent can test 12-layer models while hyperparameter agent tunes the current 11-layer config
- Quantization agent can optimize compression while evaluation agent runs multi-seed validation
- No conflicts, no waiting, no coordination overhead beyond the orchestrator

## Architecture

```
┌─────────────────────────────────────────┐
│           Research Orchestrator          │
│  (Plans iterations, dispatches agents)  │
└────────┬──────┬──────┬──────┬──────┬────┘
         │      │      │      │      │
    ┌────▼──┐ ┌─▼───┐ ┌▼────┐ ┌▼───┐ ┌▼────────┐
    │ Arch  │ │Hyper│ │Quant│ │Eval│ │Literature│
    │ Agent │ │Agent│ │Agent│ │Agt │ │  Agent   │
    └───┬───┘ └──┬──┘ └──┬──┘ └─┬──┘ └────┬─────┘
        │        │       │      │          │
    ┌───▼───┐ ┌──▼──┐ ┌──▼──┐ ┌▼──┐       │
    │Worktree│ │WT 2│ │WT 3│ │WT4│       │
    │  1     │ │     │ │    │ │   │       │
    └───┬───┘ └──┬──┘ └──┬──┘ └┬──┘       │
        └────────┴───────┴─────┘           │
                 │                         │
         ┌───────▼──────────┐    ┌─────────▼──────┐
         │ Experiment Tracker│    │  Web Search /  │
         │  (TSV + JSON)    │    │  Paper Search  │
         └──────────────────┘    └────────────────┘
```

## Components

### `src/orchestrator.py` — Research Coordinator
The brain of the system. Manages the research loop:
1. **Plan**: Decide what experiments to run based on current results and phase
2. **Execute**: Create worktrees and dispatch agents
3. **Evaluate**: Compare results, identify improvements
4. **Learn**: Update state with discoveries and failed approaches

Adaptive strategy shifts automatically:
- BPB > 1.15 → Focus on architecture (biggest gains)
- 1.13 < BPB < 1.15 → Focus on training optimization
- BPB < 1.13 → Focus on quantization (final polish)

### `src/worktree_manager.py` — Parallel Execution
Manages git worktrees for isolated parallel experiments:
- `create_worktree(name, agent_type)` → Creates branch + worktree
- `complete_worktree(name, merge=True)` → Merge improvements back to main
- `cleanup_worktree(name)` → Remove finished experiments
- `get_experiment_diff(name)` → View what an agent changed

### `src/experiment_tracker.py` — Results Database
Dual-format results tracking:
- **TSV** (human-readable, git-friendly): Quick scanning of results
- **JSON** (structured, queryable): Rich metadata for analysis

Key queries:
- `get_best(n)` → Top N experiments by BPB
- `get_by_agent(type)` → All experiments by a specific agent
- `summary()` → Overview of the entire campaign

### `src/artifact_validator.py` — Budget Enforcement
Validates the critical 16MB constraint:
- `validate_artifact_size(code_path, model_path)` → Check total size
- `estimate_model_size(n_layer, n_embd, ...)` → Predict compressed size before training
- `search_optimal_architecture(target_bytes)` → Find architectures that maximize params within budget

### `src/config.py` — Experiment Configuration
YAML-based configs with full model + training + artifact specs:
```yaml
name: sota_target
model:
  n_layer: 11
  n_embd: 512
  mlp_mult: 2.6
  unet_skip: true
training:
  optimizer: muon_adamw
  matrix_lr: 0.025
  ema_decay: 0.997
artifact:
  quantization: mixed  # int6 MLP + int8 embeddings
  compression: zlib
```

### `src/cli.py` — Command-Line Interface
```bash
python -m src.cli init <campaign_id>   # Start new research campaign
python -m src.cli plan                  # Plan next iteration
python -m src.cli status                # Current campaign status
python -m src.cli best [n]              # Top N experiments
python -m src.cli search-arch           # Find viable architectures
python -m src.cli estimate              # Estimate model size
python -m src.cli worktrees             # List active worktrees
```

## Agent Programs

Located in `agents/`, these are the autonomous loop instructions (inspired by autoresearch's `program.md`):

| Program | File | Loop |
|---------|------|------|
| Architecture search | `program_architecture.md` | Modify model structure → train → measure → keep/revert |
| Hyperparameter tuning | `program_hyperparameter.md` | Sweep one param → train → measure → keep/revert |
| Quantization optimization | `program_quantization.md` | Try quantization variant → compress → measure degradation |

Each program follows the autoresearch "NEVER STOP" principle — the agent runs experiments continuously until manually stopped.

## Experiment Configs

| Config | File | Description |
|--------|------|-------------|
| Baseline | `experiments/baseline.yaml` | Starting point: 9L/512D/2x MLP |
| SOTA Target | `experiments/sota_target.yaml` | Target: 11L/512D/2.6x MLP + innovations |

## Usage with Claude Code

The framework is designed to work with Claude Code agents defined in `.claude/agents/`:

```bash
# The orchestrator agent coordinates everything
# Architecture agent explores model changes
# Hyperparam agent tunes training
# Quantization agent optimizes compression
# Eval agent validates results
# Literature agent finds new techniques
```

Each agent runs in its own worktree, modifying `train_gpt.py` independently and reporting results back to the tracker.
