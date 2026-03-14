"""
Genetic algorithm for evolving a population of ChessTransformer models.

Generation composition
----------------------
Generation 0:         all 20 models initialised with random weights.

Subsequent generations (given scores and rankings from the tournament):
  1.  4 elites     – copies of the top-4 ranked models from the previous gen.
  2.  1 underdog   – copy of the worst-ranked model (adds diversity).
  3.  5 randoms    – freshly initialised models with random weights.
  4. 10 mutations  – each child is created by element-wise weighted crossover
                     over *all* 20 parents.  For every individual parameter
                     element, a parent is sampled with probability proportional
                     to its normalised score (score / total_score).

Population size is fixed at 20.
"""

from __future__ import annotations

import copy
import random

import torch
import torch.nn as nn

from model import ChessTransformer

POPULATION_SIZE: int = 20


# ── Random model creation ──────────────────────────────────────────────────────

def random_model(model_kwargs: dict) -> ChessTransformer:
    """Create a model and reinitialise every parameter with random normal values."""
    model = ChessTransformer(**model_kwargs)
    with torch.no_grad():
        for param in model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
    return model


# ── Element-wise weighted crossover ──────────────────────────────────────────

def _mix_parameter(
    param_list: list[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Create a new parameter tensor by sampling each element from a parent.

    For every scalar element in the parameter tensor, an independent parent is
    drawn using the provided probability weights.

    Parameters
    ----------
    param_list : list of parameter tensors with identical shape.
    weights    : 1-D float tensor of selection probabilities (sums to 1).

    Returns
    -------
    torch.Tensor with the same shape and dtype as the inputs.
    """
    # Stack into (n_parents, n_elements)
    flat = torch.stack([p.data.to(dtype=torch.float32).reshape(-1) for p in param_list])
    n_parents, n_elem = flat.shape

    # Sample a parent index for each element
    # torch.multinomial expects (n_elem, n_parents) for row-wise sampling
    parent_indices = torch.multinomial(
        weights.unsqueeze(0).expand(n_elem, n_parents),
        num_samples=1,
        replacement=True,
    ).squeeze(-1)  # (n_elem,)

    # Gather selected values: flat[parent_indices[i], i]
    result = flat[parent_indices, torch.arange(n_elem, device=flat.device)]
    return result.reshape(param_list[0].shape).to(dtype=param_list[0].dtype)


def crossover_model(
    parents: list[ChessTransformer],
    scores: list[int],
    model_kwargs: dict,
) -> ChessTransformer:
    """Create one child by element-wise weighted crossover over all parents."""
    total = sum(scores)
    weights = torch.tensor(
        [s / total for s in scores], dtype=torch.float32
    )

    child = ChessTransformer(**model_kwargs)
    param_names = [name for name, _ in parents[0].named_parameters()]

    with torch.no_grad():
        parent_param_dicts = [dict(p.named_parameters()) for p in parents]
        for name, param in child.named_parameters():
            param_list = [parent_param_dicts[i][name] for i in range(len(parents))]
            mixed = _mix_parameter(param_list, weights)
            param.data.copy_(mixed)

    return child


# ── Generation creation ────────────────────────────────────────────────────────

def first_generation(model_kwargs: dict) -> list[ChessTransformer]:
    """Create generation 0: all random models."""
    return [random_model(model_kwargs) for _ in range(POPULATION_SIZE)]


def next_generation(
    models: list[ChessTransformer],
    scores: list[int],
    rankings: list[int],
    model_kwargs: dict,
) -> list[ChessTransformer]:
    """Build the next generation from the current population.

    Parameters
    ----------
    models   : Current population (length == POPULATION_SIZE).
    scores   : Squared-rank score for each model (same indexing as models).
    rankings : Model indices sorted best-first (output of run_tournament).
    model_kwargs : Constructor kwargs for ChessTransformer.

    Returns
    -------
    New population of length POPULATION_SIZE.
    """
    new_gen: list[ChessTransformer] = []

    # 1) 4 elites: copy of top-4 from previous generation
    for i in range(4):
        new_gen.append(copy.deepcopy(models[rankings[i]]))

    # 2) 1 underdog: copy of worst-ranked model
    new_gen.append(copy.deepcopy(models[rankings[-1]]))

    # 3) 5 completely random models
    for _ in range(5):
        new_gen.append(random_model(model_kwargs))

    # 4) 10 mutation children using weighted crossover over all 20 parents
    for _ in range(10):
        new_gen.append(crossover_model(models, scores, model_kwargs))

    assert len(new_gen) == POPULATION_SIZE, (
        f"Expected {POPULATION_SIZE} children, got {len(new_gen)}"
    )
    return new_gen
