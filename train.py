"""
train.py – main entry point for the genetic chess-bot training loop.

Usage
-----
Start a fresh run (10 generations, default small model):
    python train.py --generations 10

Resume from the latest checkpoint in ./checkpoints:
    python train.py --generations 10 --resume

Resume from a specific checkpoint:
    python train.py --generations 10 --resume --checkpoint-dir ./checkpoints

Use a larger model:
    python train.py --generations 5 --d-model 128 --nhead 8 --num-layers 4 --dim-feedforward 256

Full option list:
    python train.py --help
"""

from __future__ import annotations

import argparse
import sys
import time

from genetic import first_generation, next_generation, POPULATION_SIZE
from checkpoint import save_checkpoint, load_checkpoint, latest_checkpoint_path
from tournament import run_tournament


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genetic algorithm trainer for the chess transformer bot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training
    p.add_argument("--generations",    type=int,   default=10,
                   help="Number of generations to run.")
    p.add_argument("--resume",         action="store_true",
                   help="Resume from the latest checkpoint.")
    p.add_argument("--checkpoint-dir", type=str,   default="checkpoints",
                   help="Directory for reading/writing checkpoints.")
    p.add_argument("--device",         type=str,   default="cpu",
                   help="PyTorch device (cpu / cuda / mps).")

    # Model architecture (only used when starting a *new* run)
    p.add_argument("--d-model",         type=int,   default=64,
                   help="Transformer embedding dimension.")
    p.add_argument("--nhead",           type=int,   default=4,
                   help="Number of attention heads.")
    p.add_argument("--num-layers",      type=int,   default=2,
                   help="Number of transformer encoder layers.")
    p.add_argument("--dim-feedforward", type=int,   default=128,
                   help="Feed-forward hidden size.")
    p.add_argument("--dropout",         type=float, default=0.1,
                   help="Dropout probability.")
    return p


# ── Main loop ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    device = args.device
    ckpt_dir = args.checkpoint_dir

    # ── Resume or start fresh ─────────────────────────────────────────────────
    start_gen = 0
    scores = None
    rankings = None
    wins = None

    if args.resume:
        ckpt_path = latest_checkpoint_path(ckpt_dir)
        if ckpt_path is None:
            print(
                f"[warn] --resume requested but no checkpoint found in "
                f"'{ckpt_dir}'.  Starting fresh.",
                file=sys.stderr,
            )
            args.resume = False
        else:
            print(f"[info] Resuming from {ckpt_path}")
            (start_gen, models, model_kwargs, scores, rankings, wins) = (
                load_checkpoint(ckpt_path, device=device)
            )
            start_gen += 1  # continue from next generation

    if not args.resume:
        model_kwargs = {
            "d_model":             args.d_model,
            "nhead":               args.nhead,
            "num_encoder_layers":  args.num_layers,
            "dim_feedforward":     args.dim_feedforward,
            "dropout":             args.dropout,
        }
        print("[info] Creating initial random population …")
        models = first_generation(model_kwargs)
        # Save generation 0 before evaluating so we have a checkpoint
        save_checkpoint(
            ckpt_dir, generation=0, models=models, model_kwargs=model_kwargs
        )
        print(f"[info] Generation 0 saved to '{ckpt_dir}'")

    # ── Move models to device ─────────────────────────────────────────────────
    for m in models:
        m.to(device)

    # ── Evolution loop ─────────────────────────────────────────────────────────
    end_gen = start_gen + args.generations

    for gen in range(start_gen, end_gen):
        t0 = time.time()
        n = len(models)

        print(f"\n{'='*60}")
        print(f"  Generation {gen}  –  {n*(n-1)//2} matches ({n} models)")
        print(f"{'='*60}")

        # Tournament
        scores, rankings, wins = run_tournament(models, device=device)

        elapsed = time.time() - t0
        _print_leaderboard(wins, rankings, scores, elapsed)

        # Save checkpoint with tournament results
        ckpt_path = save_checkpoint(
            ckpt_dir,
            generation=gen,
            models=models,
            model_kwargs=model_kwargs,
            scores=scores,
            rankings=rankings,
            wins=wins,
        )
        print(f"[info] Checkpoint saved: {ckpt_path}")

        # Breed next generation (skip on the last generation)
        if gen < end_gen - 1:
            models = next_generation(models, scores, rankings, model_kwargs)
            for m in models:
                m.to(device)

    print("\n[info] Training complete.")


def _print_leaderboard(
    wins: list[int],
    rankings: list[int],
    scores: list[int],
    elapsed: float,
) -> None:
    print(f"\n  Rank │ Model │ Wins │  Score  (elapsed: {elapsed:.1f}s)")
    print(f"  ─────┼───────┼──────┼────────")
    for rank, idx in enumerate(rankings, start=1):
        print(f"  {rank:4d} │  {idx:4d} │  {wins[idx]:3d} │  {scores[idx]:5d}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
