"""Research Orchestrator - Central coordination for the autonomous research loop.

Combines:
- Autoresearch's simplicity: metric-driven keep/discard, git-based tracking
- DeerFlow's sophistication: parallel agents, memory, structured decomposition

The orchestrator manages the research loop:
1. Plan: Decide what experiments to run next
2. Execute: Launch agents in parallel worktrees
3. Evaluate: Compare results, keep improvements
4. Learn: Update memory with discoveries
5. Iterate: Plan next round based on learnings
"""
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .config import ExperimentConfig
from .worktree_manager import WorktreeManager
from .experiment_tracker import ExperimentTracker, ExperimentResult
from .artifact_validator import validate_artifact_size, estimate_model_size


@dataclass
class ResearchState:
    """Current state of the research campaign."""
    campaign_id: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_best_bpb: Optional[float] = None
    current_best_experiment: Optional[str] = None
    total_experiments: int = 0
    iteration: int = 0
    # Research directions being explored
    active_directions: list[str] = field(default_factory=list)
    # Discoveries made
    discoveries: list[dict] = field(default_factory=list)
    # Failed approaches (to avoid repeating)
    failed_approaches: list[dict] = field(default_factory=list)


class ResearchOrchestrator:
    """Orchestrates the autonomous research loop for Parameter Golf.

    Usage:
        orchestrator = ResearchOrchestrator(repo_root)
        orchestrator.initialize("my_campaign")

        # Plan experiments
        plan = orchestrator.plan_iteration()

        # Execute (agents run in parallel worktrees)
        for experiment in plan:
            orchestrator.launch_experiment(experiment)

        # Evaluate and learn
        orchestrator.evaluate_iteration()
    """

    # Agent types and their roles
    AGENT_TYPES = {
        "architecture": "Explores model architecture variations (layers, attention, MLP, embeddings)",
        "hyperparameter": "Tunes training hyperparameters (LR, batch size, schedule, optimizer)",
        "quantization": "Optimizes quantization pipeline (int6/int8, GPTQ-lite, compression)",
        "evaluation": "Runs validation, checks artifact size, statistical significance",
        "literature": "Searches for new techniques from papers and repos",
    }

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.framework_dir = repo_root / "research-framework"
        self.results_dir = self.framework_dir / "results"
        self.state_file = self.framework_dir / "research_state.json"

        self.worktree_mgr = WorktreeManager(repo_root)
        self.tracker = ExperimentTracker(self.results_dir)
        self.state = self._load_state()

    def initialize(self, campaign_id: str) -> ResearchState:
        """Initialize a new research campaign."""
        self.state = ResearchState(campaign_id=campaign_id)
        self._save_state()
        return self.state

    def plan_iteration(self) -> list[dict]:
        """Plan the next iteration of experiments.

        Returns a list of experiment specs to execute.
        Strategy adapts based on current results.
        """
        self.state.iteration += 1
        best_bpb = self.tracker.get_best_bpb()
        summary = self.tracker.summary()

        experiments = []

        if self.state.iteration == 1:
            # First iteration: run baseline + initial explorations
            experiments.append({
                "name": "baseline",
                "agent_type": "evaluation",
                "description": "Run the baseline train_gpt.py to establish reference BPB",
                "priority": "high",
            })
            experiments.append({
                "name": "arch_search_depth",
                "agent_type": "architecture",
                "description": "Explore deeper architectures (10-12 layers) with adjusted MLP mult",
                "priority": "high",
            })
            experiments.append({
                "name": "arch_search_width",
                "agent_type": "architecture",
                "description": "Explore wider architectures (576-640 embd) with fewer layers",
                "priority": "medium",
            })
        else:
            # Adaptive planning based on results
            if best_bpb and best_bpb > 1.15:
                # Still far from SOTA - focus on architecture
                experiments.append({
                    "name": f"arch_iter{self.state.iteration}",
                    "agent_type": "architecture",
                    "description": "Try architectural innovations: U-Net skip, XSA, SmearGate",
                    "priority": "high",
                })
            elif best_bpb and best_bpb > 1.13:
                # Getting closer - focus on training optimization
                experiments.append({
                    "name": f"hparam_iter{self.state.iteration}",
                    "agent_type": "hyperparameter",
                    "description": "Fine-tune LR schedule, EMA decay, warmdown steps",
                    "priority": "high",
                })
            else:
                # Near SOTA - focus on quantization and compression
                experiments.append({
                    "name": f"quant_iter{self.state.iteration}",
                    "agent_type": "quantization",
                    "description": "Optimize GPTQ-lite, int6 mixed quant, compression",
                    "priority": "high",
                })

            # Always run evaluation on best config with multiple seeds
            experiments.append({
                "name": f"eval_iter{self.state.iteration}",
                "agent_type": "evaluation",
                "description": "Multi-seed validation of current best configuration",
                "priority": "high",
            })

        self._save_state()
        return experiments

    def launch_experiment(self, spec: dict) -> str:
        """Create a worktree and prepare an experiment for execution.

        Returns the worktree path where the agent should work.
        """
        wt = self.worktree_mgr.create_worktree(
            name=spec["name"],
            agent_type=spec["agent_type"],
        )

        # Create experiment record
        experiment_id = f"{spec['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = ExperimentResult(
            experiment_id=experiment_id,
            name=spec["name"],
            agent_type=spec["agent_type"],
            description=spec.get("description", ""),
            status="pending",
            branch=wt.branch,
        )
        self.tracker.record(result)
        self.state.total_experiments += 1
        self._save_state()

        return str(wt.path)

    def record_result(
        self,
        experiment_id: str,
        val_bpb: float,
        val_loss: Optional[float] = None,
        artifact_bytes: Optional[int] = None,
        training_time_sec: Optional[float] = None,
        status: str = "success",
        **kwargs,
    ) -> None:
        """Record the result of an experiment."""
        result = ExperimentResult(
            experiment_id=experiment_id,
            name=experiment_id.rsplit("_", 2)[0],
            agent_type=kwargs.get("agent_type", "unknown"),
            val_bpb=val_bpb,
            val_loss=val_loss,
            artifact_bytes=artifact_bytes,
            training_time_sec=training_time_sec,
            status=status,
        )
        self.tracker.record(result)

        # Update best if improved
        if status == "success" and val_bpb is not None:
            if self.state.current_best_bpb is None or val_bpb < self.state.current_best_bpb:
                self.state.current_best_bpb = val_bpb
                self.state.current_best_experiment = experiment_id
                self.state.discoveries.append({
                    "type": "new_best",
                    "bpb": val_bpb,
                    "experiment": experiment_id,
                    "timestamp": datetime.now().isoformat(),
                })
        self._save_state()

    def evaluate_iteration(self) -> dict:
        """Evaluate the current iteration's results and update strategy."""
        summary = self.tracker.summary()
        best = self.tracker.get_best(3)

        report = {
            "iteration": self.state.iteration,
            "summary": summary,
            "top_3": [
                {"name": r.name, "bpb": r.val_bpb, "status": r.status}
                for r in best
            ],
            "current_best_bpb": self.state.current_best_bpb,
            "leaderboard_sota": 1.1228,  # Current known SOTA
            "gap_to_sota": (
                self.state.current_best_bpb - 1.1228
                if self.state.current_best_bpb else None
            ),
        }

        return report

    def get_context_for_agent(self, agent_type: str) -> dict:
        """Get context information that should be passed to a specialized agent."""
        best = self.tracker.get_best(5)
        agent_results = self.tracker.get_by_agent(agent_type)

        return {
            "campaign_id": self.state.campaign_id,
            "iteration": self.state.iteration,
            "current_best_bpb": self.state.current_best_bpb,
            "leaderboard_sota": 1.1228,
            "top_5_experiments": [
                {"name": r.name, "bpb": r.val_bpb, "agent": r.agent_type}
                for r in best
            ],
            "agent_history": [
                {"name": r.name, "bpb": r.val_bpb, "status": r.status, "desc": r.description}
                for r in agent_results[-10:]  # Last 10 experiments by this agent type
            ],
            "failed_approaches": self.state.failed_approaches[-20:],
            "discoveries": self.state.discoveries[-10:],
        }

    def _load_state(self) -> ResearchState:
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
            return ResearchState(**data)
        return ResearchState(campaign_id="default")

    def _save_state(self) -> None:
        import dataclasses
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(dataclasses.asdict(self.state), f, indent=2, default=str)
