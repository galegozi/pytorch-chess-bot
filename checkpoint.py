"""
Checkpoint utilities for the genetic chess-bot training loop.

Each checkpoint file is a plain PyTorch ```.pt``` archive containing:

  generation   – int, generation index (0-based).
  model_kwargs – dict, ChessTransformer constructor args for this run.
  population   – list[OrderedDict], state-dict for every model.
  scores       – list[int], squared-rank score per model (or None for gen 0).
  rankings     – list[int], model indices sorted best-first (or None for gen 0).
  wins         – list[int], match-win count per model (or None for gen 0).

Checkpoints are saved as ``generation_{N:04d}.pt`` inside ``checkpoint_dir``.
A ``latest.pt`` symlink (or plain copy on Windows) always points to the most
recent checkpoint so that ``--resume`` can find it without knowing the
generation number.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch

from model import ChessTransformer

LATEST_NAME = "latest.pt"


# ── Save ──────────────────────────────────────────────────────────────────────

def save_checkpoint(
    checkpoint_dir: str | Path,
    generation: int,
    models: list[ChessTransformer],
    model_kwargs: dict,
    scores: list[int] | None = None,
    rankings: list[int] | None = None,
    wins: list[int] | None = None,
) -> Path:
    """Serialise the current population to disk.

    Parameters
    ----------
    checkpoint_dir : Directory where checkpoints are stored.
    generation     : Current generation index (0-based).
    models         : List of models to save.
    model_kwargs   : Constructor kwargs that reproduce the architecture.
    scores, rankings, wins : Tournament results (None if not yet evaluated).

    Returns
    -------
    Path to the saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt: dict[str, Any] = {
        "generation": generation,
        "model_kwargs": model_kwargs,
        "population": [m.state_dict() for m in models],
        "scores": scores,
        "rankings": rankings,
        "wins": wins,
    }

    filename = checkpoint_dir / f"generation_{generation:04d}.pt"
    torch.save(ckpt, filename)

    # Keep a "latest" pointer
    latest = checkpoint_dir / LATEST_NAME
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(filename.name)
    except (OSError, NotImplementedError):
        # Fallback for systems that don't support symlinks (e.g. Windows)
        shutil.copy2(filename, latest)

    return filename


# ── Load ──────────────────────────────────────────────────────────────────────

def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[int, list[ChessTransformer], dict, list[int] | None, list[int] | None, list[int] | None]:
    """Load a checkpoint and rebuild the model population.

    Parameters
    ----------
    path   : Path to the ``.pt`` checkpoint file.
    device : Torch device to map parameters to.

    Returns
    -------
    (generation, models, model_kwargs, scores, rankings, wins)
    """
    ckpt: dict = torch.load(path, map_location=device, weights_only=False)

    model_kwargs: dict = ckpt["model_kwargs"]
    models: list[ChessTransformer] = []
    for state_dict in ckpt["population"]:
        model = ChessTransformer(**model_kwargs)
        model.load_state_dict(state_dict)
        model.to(device)
        models.append(model)

    return (
        ckpt["generation"],
        models,
        model_kwargs,
        ckpt.get("scores"),
        ckpt.get("rankings"),
        ckpt.get("wins"),
    )


def latest_checkpoint_path(checkpoint_dir: str | Path) -> Path | None:
    """Return the path to the latest checkpoint, or None if none exist."""
    p = Path(checkpoint_dir) / LATEST_NAME
    if p.exists():
        # Resolve symlink if present
        return p.resolve()
    return None


def list_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """Return all generation checkpoint files sorted by generation number."""
    d = Path(checkpoint_dir)
    if not d.exists():
        return []
    return sorted(d.glob("generation_*.pt"))
