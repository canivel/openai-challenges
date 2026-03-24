# Architecture Research Program

You are an autonomous ML researcher exploring model architectures for the Parameter Golf challenge.
You will run experiments in a loop, modifying train_gpt.py and evaluating results.

## NEVER STOP
Run experiments continuously. After each experiment, immediately plan and start the next one.

## Workflow
1. Read train_gpt.py to understand current architecture
2. Propose ONE focused architectural change
3. Verify the change fits in the 16MB budget (estimate parameter count)
4. Commit the change with a descriptive message
5. Run: `torchrun --nproc_per_node=8 train_gpt.py > run.log 2>&1`
6. Parse results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. Log to results.tsv: commit_hash, val_bpb, vram_mb, status, description
8. Decision:
   - If val_bpb IMPROVED: keep commit, record as new baseline
   - If val_bpb WORSE or CRASHED: git reset --hard HEAD~1, log failure
9. Repeat from step 2

## Current Exploration Queue (try in order)
1. Increase depth: 9 -> 11 layers (adjust MLP mult to fit budget)
2. Add U-Net skip connections between layer i and layer (n-1-i)
3. Implement XSA (Exclusive Self Attention) on last 4 layers
4. Add SmearGate token blending
5. Implement BigramHash embeddings (2048 buckets)
6. Try partial RoPE (16 dims instead of full head_dim)
7. Add LN scale factor: 1/sqrt(layer_idx + 1)
8. Experiment with 3x MLP expansion (if budget allows)
9. Try sliding window attention pattern "SSSL"
10. Shared value embeddings across alternating layers

## Size Constraint
Every architecture change must be validated:
- Estimate compressed model size
- Code + compressed model <= 16,000,000 bytes
- If too large: reduce n_layer or mlp_mult to compensate

## Logging
Append each result to results.tsv with format:
```
commit\tval_bpb\tvram_mb\tstatus\tdescription
abc123\t1.1450\t42000\tok\t11 layers with 2.6x MLP
def456\t-\t-\tcrash\tU-Net skip OOM
```
