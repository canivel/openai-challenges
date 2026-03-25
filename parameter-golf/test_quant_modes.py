"""Quick quantization mode comparison on existing trained model.
Tests int8 vs int6 vs int6_gptq without retraining.
"""
import io
import os
import sys
import time
import zlib
import torch

# We'll import the quantization functions from train_gpt_local
# by manipulating the module directly
sys.path.insert(0, os.path.dirname(__file__))

# Load model weights
print("Loading trained model from final_model.pt ...")
state_dict = torch.load("final_model.pt", map_location="cpu", weights_only=True)
print(f"  {len(state_dict)} tensors, {sum(p.numel() for p in state_dict.values()):,} params")

# Import quantization functions
import train_gpt_local as tgl

code_bytes = 56_000  # approximate code size

results = []
for mode in ["int8", "int6", "int6_gptq"]:
    print(f"\n{'='*60}")
    print(f"  Quantization mode: {mode}")
    print(f"{'='*60}")

    # Set mode
    tgl.QUANT_MODE = mode

    t0 = time.perf_counter()
    quant_obj, stats = tgl.quantize_state_dict_int8(state_dict)
    quant_time = time.perf_counter() - t0

    # Serialize + compress
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)
    compressed_bytes = len(compressed)
    total = compressed_bytes + code_bytes

    # Dequantize and measure reconstruction error
    recon = tgl.dequantize_state_dict_int8(quant_obj)
    total_mse = 0.0
    total_elements = 0
    max_err = 0.0
    for name in recon:
        if name in state_dict:
            orig = state_dict[name].float()
            rec = recon[name].float()
            if orig.shape == rec.shape:
                diff = (orig - rec).pow(2)
                total_mse += diff.sum().item()
                total_elements += orig.numel()
                max_err = max(max_err, (orig - rec).abs().max().item())

    rmse = (total_mse / max(total_elements, 1)) ** 0.5

    fits = "YES" if total <= 16_000_000 else "NO"
    results.append({
        "mode": mode,
        "compressed_bytes": compressed_bytes,
        "total_bytes": total,
        "fits_16mb": fits,
        "rmse": rmse,
        "max_err": max_err,
        "quant_time_s": quant_time,
        "payload_bytes": stats["int8_payload_bytes"],
    })

    print(f"  Quantization time:   {quant_time:.1f}s")
    print(f"  Payload (raw):       {stats['int8_payload_bytes']:,} bytes")
    print(f"  Compressed (zlib-9): {compressed_bytes:,} bytes")
    print(f"  + code ({code_bytes:,}):   {total:,} bytes")
    print(f"  Fits 16MB budget:    {fits}")
    print(f"  Reconstruction RMSE: {rmse:.6f}")
    print(f"  Max absolute error:  {max_err:.6f}")

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"{'Mode':<12} {'Compressed':>12} {'Total':>12} {'Fits?':>6} {'RMSE':>10} {'Time':>6}")
print("-" * 60)
for r in results:
    print(f"{r['mode']:<12} {r['compressed_bytes']:>12,} {r['total_bytes']:>12,} {r['fits_16mb']:>6} {r['rmse']:>10.6f} {r['quant_time_s']:>5.1f}s")

# Size savings
if len(results) >= 2:
    base = results[0]["compressed_bytes"]
    print(f"\nSize reduction vs int8:")
    for r in results[1:]:
        saving = base - r["compressed_bytes"]
        pct = 100 * saving / base
        print(f"  {r['mode']}: {saving:+,} bytes ({pct:+.1f}%)")
