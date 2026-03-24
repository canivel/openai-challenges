"""Experiment tracking system.

Combines autoresearch's simple TSV logging with structured JSON storage
for richer experiment metadata and comparison capabilities.
"""
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_id: str
    name: str
    agent_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Metrics
    val_bpb: Optional[float] = None
    val_loss: Optional[float] = None
    artifact_bytes: Optional[int] = None
    code_bytes: Optional[int] = None
    model_bytes: Optional[int] = None
    peak_vram_mb: Optional[float] = None
    training_time_sec: Optional[float] = None
    steps_completed: Optional[int] = None
    # Config
    description: str = ""
    config_snapshot: dict = field(default_factory=dict)
    # Status
    status: str = "pending"  # pending, running, success, failed, crashed
    error_message: str = ""
    # Git
    branch: str = ""
    commit_hash: str = ""
    seed: int = 42

    @property
    def total_bytes(self) -> Optional[int]:
        if self.code_bytes is not None and self.model_bytes is not None:
            return self.code_bytes + self.model_bytes
        return self.artifact_bytes


class ExperimentTracker:
    """Tracks all experiments, results, and enables comparison.

    Stores results in both:
    - TSV file (human-readable, git-friendly, like autoresearch)
    - JSON file (structured, queryable, like deer-flow memory)
    """

    TSV_COLUMNS = [
        "experiment_id", "name", "agent_type", "timestamp",
        "val_bpb", "val_loss", "artifact_bytes", "status",
        "description", "branch", "seed",
    ]

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tsv_path = results_dir / "results.tsv"
        self.json_path = results_dir / "results.json"
        self._ensure_tsv_header()

    def _ensure_tsv_header(self) -> None:
        if not self.tsv_path.exists():
            with open(self.tsv_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(self.TSV_COLUMNS)

    def record(self, result: ExperimentResult) -> None:
        """Record an experiment result."""
        # Append to TSV
        with open(self.tsv_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([getattr(result, col, "") for col in self.TSV_COLUMNS])

        # Update JSON store
        results = self._load_json()
        results.append(asdict(result))
        self._save_json(results)

    def get_best(self, n: int = 5) -> list[ExperimentResult]:
        """Get top N experiments by val_bpb (lower is better)."""
        results = self._load_results()
        successful = [r for r in results if r.status == "success" and r.val_bpb is not None]
        successful.sort(key=lambda r: r.val_bpb)
        return successful[:n]

    def get_best_bpb(self) -> Optional[float]:
        """Get the best val_bpb achieved so far."""
        best = self.get_best(1)
        return best[0].val_bpb if best else None

    def get_by_agent(self, agent_type: str) -> list[ExperimentResult]:
        """Get all results for a specific agent type."""
        results = self._load_results()
        return [r for r in results if r.agent_type == agent_type]

    def compare(self, ids: list[str]) -> list[ExperimentResult]:
        """Compare specific experiments by ID."""
        results = self._load_results()
        return [r for r in results if r.experiment_id in ids]

    def summary(self) -> dict:
        """Get a summary of all experiments."""
        results = self._load_results()
        successful = [r for r in results if r.status == "success"]
        failed = [r for r in results if r.status in ("failed", "crashed")]
        best = self.get_best(1)
        return {
            "total_experiments": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "best_bpb": best[0].val_bpb if best else None,
            "best_experiment": best[0].name if best else None,
            "by_agent": {
                agent: len([r for r in results if r.agent_type == agent])
                for agent in set(r.agent_type for r in results)
            },
        }

    def _load_results(self) -> list[ExperimentResult]:
        data = self._load_json()
        results = []
        for d in data:
            d.pop("config_snapshot", None)  # Handle missing field gracefully
            try:
                results.append(ExperimentResult(**{k: v for k, v in d.items()
                                                   if k in ExperimentResult.__dataclass_fields__}))
            except TypeError:
                continue
        return results

    def _load_json(self) -> list[dict]:
        if self.json_path.exists():
            with open(self.json_path) as f:
                return json.load(f)
        return []

    def _save_json(self, data: list[dict]) -> None:
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
