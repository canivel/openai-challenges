"""Test Legal Score-First TTT on an already-trained model.
Loads final_model.pt, quantizes, dequantizes, then runs TTT eval.
"""
import io
import os
import sys
import time
import math
import zlib
import glob
import torch
import torch.nn.functional as F

os.environ.setdefault("TTT_ENABLED", "1")
os.environ.setdefault("TTT_EPOCHS", "1")
os.environ.setdefault("TTT_CHUNK_TOKENS", "4096")  # Small chunks for local testing
os.environ.setdefault("TTT_BATCH_SEQS", "2")
os.environ.setdefault("EVAL_STRIDE", "128")  # Larger stride = faster
os.environ.setdefault("TRAIN_SEQ_LEN", "256")
os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "1")  # Don't train, just load

sys.path.insert(0, os.path.dirname(__file__))
import train_gpt_local as tgl
from train_gpt_local import (
    Hyperparameters, GPT, eval_val_sliding_ttt,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    restore_low_dim_params_to_fp32, load_validation_tokens,
    build_sentencepiece_luts,
)

def main():
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    print("Building model...")
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        smear_gate=args.smear_gate,
        bigram_hash_buckets=args.bigram_hash_buckets,
        bigram_dim=args.bigram_dim,
        diff_attn_start=args.diff_attn_start,
    )

    # Load trained weights
    print("Loading final_model.pt...")
    state_dict = torch.load("final_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    # Quantize → dequantize (simulate roundtrip)
    print(f"Quantizing with mode={tgl.QUANT_MODE}...")
    quant_obj, _ = quantize_state_dict_int8(state_dict)
    recon = dequantize_state_dict_int8(quant_obj)
    model.load_state_dict(recon, strict=True)
    restore_low_dim_params_to_fp32(model)
    model.to(device)

    # Load validation data
    print("Loading validation tokens...")
    val_pattern = os.path.join("data", "datasets", "fineweb10B_sp1024", "fineweb_val_*.bin")
    val_tokens = load_validation_tokens(val_pattern, args.train_seq_len)
    print(f"  {val_tokens.numel():,} tokens")

    import sentencepiece as spm
    tokenizer_path = os.path.join("data", "tokenizers", "fineweb_1024_bpe.model")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # First: simple eval (no TTT) for baseline
    print("\n=== Baseline eval (no TTT, no sliding window) ===")
    model.eval()
    stride = int(os.environ.get("EVAL_STRIDE", 128))
    total_tokens = val_tokens.numel() - 1
    # Quick non-sliding eval on a small subset
    subset_seqs = min(100, total_tokens // args.train_seq_len)
    loss_sum = 0.0
    tok_count = 0
    byte_sum = 0.0
    with torch.inference_mode():
        for si in range(subset_seqs):
            start = si * args.train_seq_len
            end = start + args.train_seq_len + 1
            local = val_tokens[start:end].to(device=device, dtype=torch.int64)
            x = local[:-1].unsqueeze(0)
            y = local[1:].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y).item()
            n = args.train_seq_len
            loss_sum += loss * n
            tok_count += n
            tgt = y.reshape(-1)
            prev = x.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            byte_sum += tb.sum().item()
    baseline_loss = loss_sum / tok_count
    baseline_bpb = (baseline_loss / math.log(2.0)) * (tok_count / byte_sum)
    print(f"  Baseline val_loss={baseline_loss:.4f} val_bpb={baseline_bpb:.4f} ({subset_seqs} seqs)")

    # Now: TTT eval
    print(f"\n=== TTT eval (epochs={args.ttt_epochs}, chunk={args.ttt_chunk_tokens}, stride={stride}) ===")

    # Use only first N tokens for speed (local testing)
    max_ttt_tokens = int(os.environ.get("TTT_MAX_TOKENS", 100000))
    ttt_val_tokens = val_tokens[:min(max_ttt_tokens + 1, val_tokens.numel())]
    print(f"  Using {ttt_val_tokens.numel()-1:,} tokens for TTT test")

    t0 = time.perf_counter()
    ttt_loss, ttt_bpb = eval_val_sliding_ttt(
        args, model, 0, 1, device,
        ttt_val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, log0=print,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline (no TTT):  val_bpb = {baseline_bpb:.4f}")
    print(f"  With TTT:           val_bpb = {ttt_bpb:.4f}")
    print(f"  TTT gain:           {baseline_bpb - ttt_bpb:+.4f} BPB")
    print(f"  TTT time:           {elapsed:.1f}s")
    print(f"  Note: Tested on first {max_ttt_tokens:,} tokens only (local speed)")

if __name__ == "__main__":
    main()
