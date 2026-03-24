"""Git worktree manager for parallel experiment execution.

Inspired by autoresearch's git-based experiment tracking, but extended
to support parallel experiments via git worktrees. Each experiment
runs in its own worktree with an isolated branch, enabling multiple
agents to work simultaneously without conflicts.
"""
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Worktree:
    """Represents a git worktree for an experiment."""
    name: str
    path: Path
    branch: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"  # active, completed, failed, abandoned
    experiment_id: Optional[str] = None
    agent_type: Optional[str] = None


class WorktreeManager:
    """Manages git worktrees for parallel agent experiments.

    Each agent (architecture explorer, hyperparam tuner, etc.) gets its own
    worktree so they can modify train_gpt.py independently and run experiments
    in parallel on different GPU allocations.
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.worktrees_dir = repo_root / ".worktrees"
        self.state_file = repo_root / ".worktrees" / "state.json"
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

    def _run_git(self, *args: str, cwd: Optional[Path] = None) -> str:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=cwd or self.repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
        return result.stdout.strip()

    def create_worktree(
        self,
        name: str,
        agent_type: str,
        base_branch: str = "main",
    ) -> Worktree:
        """Create a new worktree for an experiment agent.

        Args:
            name: Human-readable name for the experiment
            agent_type: Type of agent (architecture, hyperparam, quantization, eval)
            base_branch: Branch to base the worktree on

        Returns:
            Worktree object with path and branch info
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"experiment/{agent_type}/{name}_{timestamp}"
        worktree_path = self.worktrees_dir / f"{agent_type}_{name}_{timestamp}"

        # Create branch and worktree
        self._run_git("branch", branch_name, base_branch)
        self._run_git("worktree", "add", str(worktree_path), branch_name)

        wt = Worktree(
            name=name,
            path=worktree_path,
            branch=branch_name,
            agent_type=agent_type,
        )
        self._save_worktree(wt)
        return wt

    def list_worktrees(self) -> list[Worktree]:
        """List all managed worktrees."""
        state = self._load_state()
        return [Worktree(**wt) for wt in state.get("worktrees", [])]

    def get_active_worktrees(self) -> list[Worktree]:
        """Get worktrees that are currently in use."""
        return [wt for wt in self.list_worktrees() if wt.status == "active"]

    def complete_worktree(self, name: str, merge: bool = False) -> None:
        """Mark a worktree as completed, optionally merging results."""
        state = self._load_state()
        for wt_data in state.get("worktrees", []):
            if wt_data["name"] == name:
                wt_data["status"] = "completed"
                if merge:
                    self._run_git("checkout", "main")
                    self._run_git("merge", wt_data["branch"], "--no-ff",
                                  "-m", f"Merge experiment: {name}")
                break
        self._save_state(state)

    def cleanup_worktree(self, name: str) -> None:
        """Remove a worktree and its branch."""
        state = self._load_state()
        for i, wt_data in enumerate(state.get("worktrees", [])):
            if wt_data["name"] == name:
                path = Path(wt_data["path"])
                branch = wt_data["branch"]
                if path.exists():
                    self._run_git("worktree", "remove", str(path), "--force")
                try:
                    self._run_git("branch", "-D", branch)
                except RuntimeError:
                    pass  # Branch may already be deleted
                state["worktrees"].pop(i)
                break
        self._save_state(state)

    def get_experiment_diff(self, name: str) -> str:
        """Get the diff of changes made in a worktree vs main."""
        worktrees = self.list_worktrees()
        for wt in worktrees:
            if wt.name == name:
                return self._run_git("diff", "main...", wt.branch)
        raise ValueError(f"Worktree {name} not found")

    def _save_worktree(self, wt: Worktree) -> None:
        state = self._load_state()
        if "worktrees" not in state:
            state["worktrees"] = []
        state["worktrees"].append({
            "name": wt.name,
            "path": str(wt.path),
            "branch": wt.branch,
            "created_at": wt.created_at,
            "status": wt.status,
            "experiment_id": wt.experiment_id,
            "agent_type": wt.agent_type,
        })
        self._save_state(state)

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"worktrees": []}

    def _save_state(self, state: dict) -> None:
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
