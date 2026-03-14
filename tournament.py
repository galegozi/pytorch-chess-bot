"""
Tournament module – round-robin evaluation of a population of chess bots.

Game flow
---------
``play_game``   – one game, returns result from white's perspective.
``play_match``  – best-of-3 match between two models.
``run_tournament`` – full round-robin; returns scores and rankings.

Scoring
-------
After ranking all models by match wins (most wins = rank 1), each model
receives a score equal to the square of its rank *position from the top*:

  rank 1 (best)  → n²   (e.g. 400 for n = 20)
  rank 2         → (n-1)²
  …
  rank n (worst) → 1

where *n* is the population size.

Draw handling
-------------
Games that are not finished after ``MAX_MOVES`` half-moves are scored as draws.
"""

from __future__ import annotations

import random

import chess

from chess_utils import encode_board, legal_moves_mask, index_to_move
from model import ChessTransformer

# Hard cap on half-moves per game to avoid infinite loops
MAX_MOVES: int = 400


# ── Single game ───────────────────────────────────────────────────────────────

def play_game(
    model_white: ChessTransformer,
    model_black: ChessTransformer,
    device: str = "cpu",
) -> float:
    """Play one chess game between two models.

    Parameters
    ----------
    model_white, model_black:
        The models controlling the white and black pieces respectively.
    device:
        Torch device string.

    Returns
    -------
    float
        ``1.0`` – white wins, ``0.0`` – black wins, ``0.5`` – draw / timeout.
    """
    board = chess.Board()

    for _ in range(MAX_MOVES):
        if board.is_game_over(claim_draw=True):
            break

        model = model_white if board.turn == chess.WHITE else model_black
        board_tokens, extra_tokens = encode_board(board)
        mask = legal_moves_mask(board)

        # Add batch dimension
        bt = board_tokens.unsqueeze(0).to(device)
        et = extra_tokens.unsqueeze(0).to(device)
        lm = mask.unsqueeze(0).to(device)

        move_idx = model.select_move(bt, et, lm).item()
        move = index_to_move(move_idx, board)

        if move in board.legal_moves:
            board.push(move)
        else:
            # Fallback: random legal move (should not normally happen)
            board.push(random.choice(list(board.legal_moves)))

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.5  # timeout → draw
    if outcome.winner is None:
        return 0.5  # draw
    return 1.0 if outcome.winner == chess.WHITE else 0.0


# ── Best-of-3 match ───────────────────────────────────────────────────────────

def play_match(
    model_a: ChessTransformer,
    model_b: ChessTransformer,
    device: str = "cpu",
) -> tuple[int, int]:
    """Play a best-of-3 match between model_a and model_b.

    Game 1: model_a = white.
    Game 2: model_b = white.
    Game 3 (if 1–1 after two games): randomly assigned colours.

    Returns
    -------
    (wins_a, wins_b) – each value is the number of decisive games won.
    """
    wins_a = 0
    wins_b = 0

    # Game 1: A = white
    result = play_game(model_a, model_b, device)
    if result == 1.0:
        wins_a += 1
    elif result == 0.0:
        wins_b += 1

    # Game 2: B = white
    result = play_game(model_b, model_a, device)
    if result == 1.0:
        wins_b += 1
    elif result == 0.0:
        wins_a += 1

    # Game 3 only if still tied
    if wins_a == wins_b:
        if random.random() < 0.5:
            # A = white
            result = play_game(model_a, model_b, device)
            if result == 1.0:
                wins_a += 1
            elif result == 0.0:
                wins_b += 1
        else:
            # B = white
            result = play_game(model_b, model_a, device)
            if result == 1.0:
                wins_b += 1
            elif result == 0.0:
                wins_a += 1

    return wins_a, wins_b


# ── Round-robin tournament ────────────────────────────────────────────────────

def run_tournament(
    models: list[ChessTransformer],
    device: str = "cpu",
) -> tuple[list[int], list[int], list[int]]:
    """Run a full round-robin tournament (every pair plays once).

    Parameters
    ----------
    models:
        Population of models; no model plays itself.
    device:
        Torch device string.

    Returns
    -------
    scores   : list[int]  – squared-rank score for each model.
    rankings : list[int]  – model indices sorted best → worst.
    wins     : list[int]  – total match wins per model.
    """
    n = len(models)
    wins = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            wins_i, wins_j = play_match(models[i], models[j], device)
            wins[i] += wins_i
            wins[j] += wins_j

    # Sort indices by total wins (descending)
    rankings = sorted(range(n), key=lambda k: wins[k], reverse=True)

    # Assign squared-rank scores: best gets n², worst gets 1²
    scores = [0] * n
    for rank, model_idx in enumerate(rankings):
        scores[model_idx] = (n - rank) ** 2

    return scores, rankings, wins
