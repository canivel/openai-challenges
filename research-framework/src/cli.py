"""CLI entry point for the research framework."""
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    from .orchestrator import ResearchOrchestrator
    from .artifact_validator import search_optimal_architecture, estimate_model_size

    repo_root = Path(__file__).parent.parent.parent

    if len(sys.argv) < 2:
        print("Golf Research Framework - Autonomous ML Research for Parameter Golf")
        print()
        print("Commands:")
        print("  init <campaign_id>     Initialize a new research campaign")
        print("  plan                   Plan next iteration of experiments")
        print("  status                 Show current research status")
        print("  best [n]               Show top N experiments")
        print("  search-arch            Search for optimal architecture configs")
        print("  estimate               Estimate model size for given config")
        print("  worktrees              List active worktrees")
        return

    cmd = sys.argv[1]
    orchestrator = ResearchOrchestrator(repo_root)

    if cmd == "init":
        campaign_id = sys.argv[2] if len(sys.argv) > 2 else "default"
        state = orchestrator.initialize(campaign_id)
        print(f"Initialized campaign: {state.campaign_id}")

    elif cmd == "plan":
        plan = orchestrator.plan_iteration()
        print(f"Iteration {orchestrator.state.iteration} plan:")
        for exp in plan:
            print(f"  [{exp['agent_type']}] {exp['name']}: {exp['description']}")

    elif cmd == "status":
        report = orchestrator.evaluate_iteration()
        print(f"Iteration: {report['iteration']}")
        print(f"Best BPB: {report['current_best_bpb']}")
        print(f"SOTA: {report['leaderboard_sota']}")
        print(f"Gap: {report['gap_to_sota']}")
        print(f"Summary: {report['summary']}")

    elif cmd == "best":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        best = orchestrator.tracker.get_best(n)
        for r in best:
            print(f"  {r.val_bpb:.4f} BPB | {r.name} ({r.agent_type}) | {r.status}")

    elif cmd == "search-arch":
        configs = search_optimal_architecture()
        print(f"Top {len(configs)} viable architectures (sorted by param count):")
        for c in configs[:10]:
            print(f"  L={c['n_layer']} D={c['n_embd']} H={c['n_head']} "
                  f"KV={c['n_kv_head']} MLP={c['mlp_mult']:.1f}x "
                  f"| {c['total_params']:,} params "
                  f"| ~{c['estimated_compressed_bytes']:,} bytes")

    elif cmd == "worktrees":
        wts = orchestrator.worktree_mgr.list_worktrees()
        if not wts:
            print("No active worktrees")
        for wt in wts:
            print(f"  [{wt.status}] {wt.name} ({wt.agent_type}) @ {wt.branch}")

    elif cmd == "estimate":
        est = estimate_model_size(
            n_layer=11, n_embd=512, n_head=8, n_kv_head=4,
            mlp_mult=2.6, vocab_size=1024, tied_embeddings=True, quant_bits=8,
        )
        print("Model size estimate (11L/512D/2.6x MLP):")
        for k, v in est.items():
            if isinstance(v, int) and v > 1000:
                print(f"  {k}: {v:,}")
            else:
                print(f"  {k}: {v}")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
