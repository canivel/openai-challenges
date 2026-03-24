"""Configuration system for the research framework."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class GPUConfig:
    num_gpus: int = 8
    gpu_type: str = "H100"
    max_training_time_sec: int = 600  # 10 minutes
    max_eval_time_sec: int = 600


@dataclass
class ArtifactConfig:
    max_bytes: int = 16_000_000  # 16MB decimal
    quantization: str = "int8"  # int6, int8, mixed
    compression: str = "zlib"  # zlib, zstd
    compression_level: int = 9


@dataclass
class ModelConfig:
    n_layer: int = 11
    n_embd: int = 512
    n_head: int = 8
    n_kv_head: int = 4
    mlp_mult: float = 2.6
    vocab_size: int = 1024
    seq_len: int = 1024
    tied_embeddings: bool = True


@dataclass
class TrainingConfig:
    optimizer: str = "muon_adamw"
    matrix_lr: float = 0.025
    scalar_lr: float = 0.025
    weight_decay: float = 0.04
    warmup_steps: int = 20
    warmdown_steps: int = 3500
    batch_tokens: int = 524_288
    grad_clip: float = 0.3
    ema_decay: float = 0.997
    use_ema: bool = True
    use_swa: bool = True
    seeds: list[int] = field(default_factory=lambda: [42, 1337, 2024])


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    description: str = ""
    gpu: GPUConfig = field(default_factory=GPUConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Paths
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    parameter_golf_dir: Path = field(default_factory=lambda: Path.cwd() / "parameter-golf")
    results_dir: Path = field(default_factory=lambda: Path.cwd() / "research-framework" / "results")

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        gpu = GPUConfig(**data.get("gpu", {}))
        artifact = ArtifactConfig(**data.get("artifact", {}))
        model = ModelConfig(**data.get("model", {}))
        training = TrainingConfig(**data.get("training", {}))
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            gpu=gpu,
            artifact=artifact,
            model=model,
            training=training,
        )

    def to_yaml(self, path: Path) -> None:
        import dataclasses
        data = dataclasses.asdict(self)
        # Convert Path objects to strings
        for key in ("repo_root", "parameter_golf_dir", "results_dir"):
            data[key] = str(data[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
