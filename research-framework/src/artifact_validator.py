"""Artifact validation and quantization utilities.

Validates that model artifacts meet Parameter Golf constraints:
- Total artifact (code + compressed model) <= 16,000,000 bytes
- Supports int6, int8, and mixed quantization
- Compression via zlib or zstd
"""
import struct
import zlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


MAX_ARTIFACT_BYTES = 16_000_000  # 16MB decimal, NOT 16 MiB


@dataclass
class ArtifactReport:
    """Report on artifact size and validity."""
    code_bytes: int
    model_bytes_raw: int
    model_bytes_compressed: int
    total_bytes: int
    is_valid: bool
    budget_remaining: int
    quantization_type: str
    compression_type: str
    num_parameters: int = 0
    # Per-component breakdown
    embedding_bytes: int = 0
    attention_bytes: int = 0
    mlp_bytes: int = 0
    norm_bytes: int = 0
    other_bytes: int = 0


def validate_artifact_size(
    code_path: Path,
    model_path: Optional[Path] = None,
    model_bytes_compressed: Optional[int] = None,
) -> ArtifactReport:
    """Validate that code + model fits within 16MB budget.

    Args:
        code_path: Path to train_gpt.py
        model_path: Path to compressed model file (if saved)
        model_bytes_compressed: Pre-computed compressed size (alternative to model_path)
    """
    code_bytes = code_path.stat().st_size

    if model_path and model_path.exists():
        model_compressed = model_path.stat().st_size
        model_raw = model_compressed  # Already compressed on disk
    elif model_bytes_compressed is not None:
        model_compressed = model_bytes_compressed
        model_raw = model_bytes_compressed
    else:
        model_compressed = 0
        model_raw = 0

    total = code_bytes + model_compressed
    return ArtifactReport(
        code_bytes=code_bytes,
        model_bytes_raw=model_raw,
        model_bytes_compressed=model_compressed,
        total_bytes=total,
        is_valid=total <= MAX_ARTIFACT_BYTES,
        budget_remaining=MAX_ARTIFACT_BYTES - total,
        quantization_type="unknown",
        compression_type="unknown",
    )


def estimate_model_size(
    n_layer: int,
    n_embd: int,
    n_head: int,
    n_kv_head: int,
    mlp_mult: float,
    vocab_size: int,
    tied_embeddings: bool = True,
    quant_bits: int = 8,
) -> dict:
    """Estimate compressed model size given architecture parameters.

    Returns breakdown of parameter counts and estimated compressed bytes.
    """
    head_dim = n_embd // n_head

    # Embedding
    emb_params = vocab_size * n_embd
    if not tied_embeddings:
        emb_params *= 2  # Separate input/output embeddings

    # Per-layer attention
    q_params = n_embd * n_embd  # Q projection
    k_params = n_embd * (n_kv_head * head_dim)  # K projection (GQA)
    v_params = n_embd * (n_kv_head * head_dim)  # V projection (GQA)
    o_params = n_embd * n_embd  # Output projection
    attn_params_per_layer = q_params + k_params + v_params + o_params

    # Per-layer MLP
    mlp_hidden = int(n_embd * mlp_mult)
    mlp_params_per_layer = 2 * n_embd * mlp_hidden  # Up + down projections (gate folded in)

    # Per-layer norms (RMSNorm - just scale vectors)
    norm_params_per_layer = 2 * n_embd  # attention norm + mlp norm

    # Total
    total_attn = attn_params_per_layer * n_layer
    total_mlp = mlp_params_per_layer * n_layer
    total_norm = norm_params_per_layer * n_layer
    total_params = emb_params + total_attn + total_mlp + total_norm

    # Estimate compressed size
    # int8: ~1 byte per param + scales overhead
    # int6: ~0.75 bytes per param + scales overhead
    # zlib typically achieves ~0.85-0.95 compression ratio on quantized weights
    bytes_per_param = quant_bits / 8
    scale_overhead = 0.02  # ~2% overhead for quantization scales
    compression_ratio = 0.92  # zlib level 9 on quantized data

    raw_bytes = int(total_params * bytes_per_param * (1 + scale_overhead))
    compressed_bytes = int(raw_bytes * compression_ratio)

    return {
        "total_params": total_params,
        "embedding_params": emb_params,
        "attention_params": total_attn,
        "mlp_params": total_mlp,
        "norm_params": total_norm,
        "estimated_raw_bytes": raw_bytes,
        "estimated_compressed_bytes": compressed_bytes,
        "bytes_per_param": bytes_per_param,
        "fits_budget": compressed_bytes <= (MAX_ARTIFACT_BYTES - 70_000),  # ~70KB code budget
    }


def search_optimal_architecture(
    target_bytes: int = 15_500_000,  # Leave room for code
    quant_bits: int = 8,
) -> list[dict]:
    """Search for architecture configs that fit within byte budget.

    Returns list of viable configurations sorted by parameter count (more is better).
    """
    configs = []

    for n_layer in range(8, 14):
        for mlp_mult_x10 in range(20, 35):  # 2.0 to 3.4
            mlp_mult = mlp_mult_x10 / 10.0
            for n_embd in [384, 448, 512, 576]:
                for n_kv_head in [2, 4]:
                    n_head = 8 if n_embd >= 512 else (n_embd // 64)
                    if n_embd % n_head != 0:
                        continue

                    est = estimate_model_size(
                        n_layer=n_layer,
                        n_embd=n_embd,
                        n_head=n_head,
                        n_kv_head=n_kv_head,
                        mlp_mult=mlp_mult,
                        vocab_size=1024,
                        tied_embeddings=True,
                        quant_bits=quant_bits,
                    )

                    if est["estimated_compressed_bytes"] <= target_bytes:
                        configs.append({
                            "n_layer": n_layer,
                            "n_embd": n_embd,
                            "n_head": n_head,
                            "n_kv_head": n_kv_head,
                            "mlp_mult": mlp_mult,
                            **est,
                        })

    # Sort by total params descending (more params = potentially better)
    configs.sort(key=lambda c: c["total_params"], reverse=True)
    return configs[:20]  # Top 20 candidates
