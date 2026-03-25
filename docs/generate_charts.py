"""Generate charts for the Parameter Golf white paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# ========== Figure 1: Training Loss Curves ==========
fig, ax = plt.subplots(figsize=(10, 6))

# v1 data
v1_steps = [0, 1, 10, 25, 50, 75, 100, 118]
v1_loss =  [6.9357, 6.9378, 7.6557, 5.6428, 5.3553, 5.0759, 4.8789, 4.7592]

# v2 data
v2_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 127]
v2_loss =  [6.9337, 6.9329, 17.1242, 16.9123, 16.6471, 16.3132, 15.8925, 15.5388,
            15.1488, 14.8106, 14.2310, 9.9742, 7.2547, 5.9852, 5.6416, 5.5286,
            5.4443, 5.4597, 5.3215, 5.3594, 5.2876, 5.2835, 5.3259]

ax.plot(v1_steps, v1_loss, 'b-o', markersize=5, label='Baseline v1 (17M, seq=512)', linewidth=2)
ax.plot(v2_steps, v2_loss, 'r-s', markersize=4, label='Improved v2 (26.8M, seq=256)', linewidth=2)
ax.set_xlabel('Training Step')
ax.set_ylabel('Loss (Cross-Entropy)')
ax.set_title('Training Loss: Baseline vs. Improved Architecture')
ax.legend()
ax.set_ylim(3, 18)
ax.set_xlim(-2, 135)
fig.tight_layout()
fig.savefig('docs/fig1_training_loss.png')
print("Saved fig1_training_loss.png")

# ========== Figure 2: Validation BPB ==========
fig, ax = plt.subplots(figsize=(10, 6))

v1_val_steps = [0, 100, 118]
v1_val_bpb =   [4.1077, 2.8524, 2.8187]

v2_val_steps = [0, 50, 100, 127]
v2_val_bpb =   [4.1065, 3.3306, 3.1653, 3.1543]

ax.plot(v1_val_steps, v1_val_bpb, 'b-o', markersize=8, label='Baseline v1 (17M, seq=512)', linewidth=2)
ax.plot(v2_val_steps, v2_val_bpb, 'r-s', markersize=8, label='Improved v2 (26.8M, seq=256)', linewidth=2)

# Add SOTA reference line
ax.axhline(y=1.1228, color='green', linestyle='--', alpha=0.7, label='SOTA (1.1228 BPB)')
ax.axhline(y=1.2243, color='orange', linestyle='--', alpha=0.7, label='Baseline full (1.2243 BPB)')

ax.set_xlabel('Training Step')
ax.set_ylabel('Validation BPB (lower is better)')
ax.set_title('Validation BPB Progression')
ax.legend()
ax.set_ylim(0.8, 4.5)
fig.tight_layout()
fig.savefig('docs/fig2_val_bpb.png')
print("Saved fig2_val_bpb.png")

# ========== Figure 3: Artifact Budget ==========
fig, ax = plt.subplots(figsize=(10, 5))

categories = ['Baseline v1\n(int8)', 'Improved v2\n(int8)', 'SOTA\n(int6+GPTQ)', 'Budget\nLimit']
sizes_mb = [6.35, 8.95, 15.50, 16.00]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

bars = ax.barh(categories, sizes_mb, color=colors, edgecolor='white', height=0.6)
ax.axvline(x=16.0, color='red', linestyle='--', linewidth=2, label='16 MB limit')

for bar, size in zip(bars, sizes_mb):
    pct = size / 16.0 * 100
    ax.text(size + 0.2, bar.get_y() + bar.get_height()/2,
            f'{size:.2f} MB ({pct:.0f}%)', va='center', fontweight='bold')

ax.set_xlabel('Artifact Size (MB)')
ax.set_title('Artifact Budget Utilization')
ax.set_xlim(0, 20)
ax.legend()
fig.tight_layout()
fig.savefig('docs/fig3_artifact_budget.png')
print("Saved fig3_artifact_budget.png")

# ========== Figure 4: Technique Contributions ==========
fig, ax = plt.subplots(figsize=(12, 7))

techniques = [
    'Sliding window eval (s=64)',
    'Sequence length 2048',
    'Int6 quant + 3x MLP',
    'Depth (10-11 layers)',
    'SmearGate + BigramHash',
    'XSA (last 4 layers)',
    'Partial RoPE (16d)',
    'FP16 embeddings',
    'EMA (decay=0.997)',
    'GPTQ-lite optimization',
    'Warmdown tuning (3500)',
    'Late QAT (4%)',
]
contributions = [0.0335, 0.0186, 0.0100, 0.0050, 0.0050, 0.0023, 0.0023, 0.0020, 0.0006, 0.0006, 0.0002, 0.0001]
implemented = [False, False, True, True, True, True, True, False, True, False, True, False]

colors = ['#55A868' if imp else '#C44E52' for imp in implemented]

bars = ax.barh(techniques[::-1], contributions[::-1], color=colors[::-1], edgecolor='white', height=0.6)
ax.set_xlabel('BPB Improvement (lower = better)')
ax.set_title('Individual Technique Contributions to BPB')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#55A868', label='Implemented'),
                   Patch(facecolor='#C44E52', label='Planned')]
ax.legend(handles=legend_elements, loc='lower right')

for bar, val in zip(bars, contributions[::-1]):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f'-{val:.4f}', va='center', fontsize=9)

fig.tight_layout()
fig.savefig('docs/fig4_technique_contributions.png')
print("Saved fig4_technique_contributions.png")

# ========== Figure 5: Projected BPB Waterfall ==========
fig, ax = plt.subplots(figsize=(12, 6))

stages = ['Baseline', '+Sliding\nEval', '+11L/3x\nMLP', '+XSA\n+RoPE16\n+LNS',
          '+Smear\n+Bigram', '+Seq\n2048', '+Int6\n+GPTQ', '+EMA\n+SWA', '+Tuning']
values = [1.2243, 1.1908, 1.1758, 1.1692, 1.1642, 1.1456, 1.1350, 1.1264, 1.1228]

# Colors: implemented=green, planned=red
stage_impl = [True, False, True, True, True, False, False, True, True]
colors = ['#55A868' if imp else '#DD8452' for imp in stage_impl]

ax.bar(range(len(stages)), values, color=colors, edgecolor='white', width=0.7)
ax.axhline(y=1.1228, color='green', linestyle='--', alpha=0.5, label='SOTA (1.1228)')
ax.axhline(y=1.2243, color='red', linestyle='--', alpha=0.3, label='Baseline (1.2243)')

ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel('Projected BPB')
ax.set_title('Projected BPB Improvement Waterfall')
ax.set_ylim(1.08, 1.26)
ax.legend()

for i, (v, s) in enumerate(zip(values, stages)):
    ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')

fig.tight_layout()
fig.savefig('docs/fig5_bpb_waterfall.png')
print("Saved fig5_bpb_waterfall.png")

# ========== Figure 6: Parameter Budget Allocation ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# v1
labels_v1 = ['Attention\n(Q,K,V,O)', 'MLP\n(fc+proj)', 'Embedding', 'Other']
sizes_v1 = [9437184, 4718592, 524288, 17059912-9437184-4718592-524288]
ax1.pie(sizes_v1, labels=labels_v1, autopct='%1.1f%%', colors=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
ax1.set_title(f'Baseline v1\n({sum(sizes_v1)/1e6:.1f}M params)')

# v2
labels_v2 = ['Attention\n(Q,K,V,O)', 'MLP\n(fc+proj)', 'Embedding', 'BigramHash', 'Other']
sizes_v2 = [12582912, 10616832, 524288, 524288, 26829912-12582912-10616832-524288-524288]
ax2.pie(sizes_v2, labels=labels_v2, autopct='%1.1f%%', colors=['#4C72B0', '#DD8452', '#55A868', '#8172B3', '#C44E52'])
ax2.set_title(f'Improved v2\n({sum(sizes_v2)/1e6:.1f}M params)')

fig.suptitle('Parameter Budget Allocation', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('docs/fig6_param_allocation.png')
print("Saved fig6_param_allocation.png")

# ========== Figure 7: Quantization Degradation ==========
fig, ax = plt.subplots(figsize=(10, 5))

methods = ['Our v2\n(int8, no GPTQ)', 'SOTA\n(int6, GPTQ)', 'SOTA\n(int6, GPTQ+QAT)']
degradation = [0.3124, 0.0190, 0.0001]
colors = ['#C44E52', '#DD8452', '#55A868']

bars = ax.bar(methods, degradation, color=colors, edgecolor='white', width=0.5)
ax.set_ylabel('BPB Degradation (lower is better)')
ax.set_title('Quantization Degradation: Post-Quant BPB - Pre-Quant BPB')
ax.set_yscale('log')
ax.set_ylim(0.00005, 1.0)

for bar, val in zip(bars, degradation):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.3,
            f'{val:.4f}', ha='center', fontweight='bold')

fig.tight_layout()
fig.savefig('docs/fig7_quant_degradation.png')
print("Saved fig7_quant_degradation.png")

print("\nAll charts generated successfully!")
