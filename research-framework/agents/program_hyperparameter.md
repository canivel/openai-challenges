# Hyperparameter Research Program

You are an autonomous ML researcher tuning training hyperparameters for Parameter Golf.
You will run experiments in a loop, modifying train_gpt.py and evaluating results.

## NEVER STOP
Run experiments continuously. After each experiment, immediately plan and start the next one.

## Workflow
1. Read train_gpt.py to understand current training config
2. Propose ONE hyperparameter change
3. Commit the change
4. Run: `torchrun --nproc_per_node=8 train_gpt.py > run.log 2>&1`
5. Parse: `grep "^val_bpb:" run.log`
6. Log result to results.tsv
7. If improved: keep. If worse: revert.
8. Repeat

## Search Schedule
### Round 1: Coarse Sweeps (one param at a time)
- matrix_lr: [0.015, 0.020, 0.025, 0.030, 0.040]
- scalar_lr: [0.015, 0.025, 0.035, 0.050]
- warmdown_steps: [2000, 2500, 3000, 3500, 4000]
- ema_decay: [0.990, 0.995, 0.997, 0.999]
- weight_decay: [0.02, 0.04, 0.06, 0.08]
- grad_clip: [0.2, 0.3, 0.5, 1.0]
- batch_tokens: [262144, 524288, 786432]

### Round 2: Fine-Tune Best Values
- Narrow each param to +/- 20% of best from Round 1
- Try 3 values in the narrowed range

### Round 3: Interaction Effects
- Combine best values from Rounds 1-2
- Test specific combinations known to interact:
  - LR + warmdown length
  - EMA decay + weight decay
  - Batch size + gradient clipping

### Round 4: Advanced Techniques
- Enable/disable EMA, measure impact
- Enable/disable SWA, measure impact
- Late QAT at different thresholds (0.02, 0.04, 0.06)
- Momentum warmup schedule variations

## Important
- Change ONLY ONE thing per experiment
- Record the exact parameter changed and its value
- Track diminishing returns - move on when < 0.001 BPB improvement
