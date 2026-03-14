"""
Unit tests for the pytorch-chess-bot implementation.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import chess
import pytest
import torch

# ── Module imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import ChessTransformer, build_model
from chess_utils import (
    encode_board,
    legal_moves_mask,
    move_to_index,
    index_to_move,
    NUM_MOVE_INDICES,
)
from genetic import (
    random_model,
    crossover_model,
    first_generation,
    next_generation,
    POPULATION_SIZE,
)
from checkpoint import save_checkpoint, load_checkpoint, latest_checkpoint_path
from tournament import play_game, play_match, run_tournament


# ── Fixtures ──────────────────────────────────────────────────────────────────

SMALL_KWARGS = {
    "d_model": 16,
    "nhead": 2,
    "num_encoder_layers": 1,
    "dim_feedforward": 32,
    "dropout": 0.0,
}


@pytest.fixture
def small_model():
    return ChessTransformer(**SMALL_KWARGS)


@pytest.fixture
def start_board():
    return chess.Board()


# ══════════════════════════════════════════════════════════════════════════════
# model.py
# ══════════════════════════════════════════════════════════════════════════════

class TestChessTransformer:
    def test_default_construction(self):
        model = ChessTransformer()
        assert model.d_model == 64
        assert model.num_moves == 4096

    def test_small_construction(self, small_model):
        assert small_model.d_model == 16

    def test_forward_pass_shape(self, small_model):
        board_tokens = torch.zeros(2, 64, dtype=torch.long)
        extra_tokens = torch.zeros(2, 6, dtype=torch.long)
        logits = small_model(board_tokens, extra_tokens)
        assert logits.shape == (2, 4096)

    def test_select_move_returns_legal(self, small_model, start_board):
        board_tokens, extra_tokens = encode_board(start_board)
        mask = legal_moves_mask(start_board)
        bt = board_tokens.unsqueeze(0)
        et = extra_tokens.unsqueeze(0)
        lm = mask.unsqueeze(0)
        move_idx = small_model.select_move(bt, et, lm).item()
        assert mask[move_idx].item(), "Selected move must be legal"

    def test_get_config_roundtrip(self, small_model):
        cfg = small_model.get_config()
        rebuilt = ChessTransformer(**cfg)
        assert rebuilt.d_model == small_model.d_model
        assert rebuilt.num_encoder_layers == small_model.num_encoder_layers

    def test_parameter_count_small(self, small_model):
        total = sum(p.numel() for p in small_model.parameters())
        # Ensure the small model has fewer parameters than the default
        default = ChessTransformer()
        default_total = sum(p.numel() for p in default.parameters())
        assert total < default_total


# ══════════════════════════════════════════════════════════════════════════════
# chess_utils.py
# ══════════════════════════════════════════════════════════════════════════════

class TestChessUtils:
    def test_encode_board_shapes(self, start_board):
        board_tokens, extra_tokens = encode_board(start_board)
        assert board_tokens.shape == (64,)
        assert extra_tokens.shape == (6,)

    def test_encode_board_dtypes(self, start_board):
        bt, et = encode_board(start_board)
        assert bt.dtype == torch.long
        assert et.dtype == torch.long

    def test_piece_types_in_range(self, start_board):
        bt, _ = encode_board(start_board)
        assert bt.min().item() >= 0
        assert bt.max().item() <= 12

    def test_extra_tokens_in_range(self, start_board):
        _, et = encode_board(start_board)
        # All values should fit into the extra_embedding vocabulary (< 128)
        assert et.min().item() >= 0
        assert et.max().item() < 128

    def test_legal_moves_mask_starting_position(self, start_board):
        mask = legal_moves_mask(start_board)
        assert mask.shape == (NUM_MOVE_INDICES,)
        assert mask.dtype == torch.bool
        # Starting position has 20 legal moves
        assert mask.sum().item() == 20

    def test_legal_moves_mask_empty_board(self):
        board = chess.Board(fen=None)  # empty board
        mask = legal_moves_mask(board)
        assert mask.sum().item() == 0

    def test_move_roundtrip(self, start_board):
        for move in start_board.legal_moves:
            idx = move_to_index(move)
            recovered = index_to_move(idx, start_board)
            assert recovered.from_square == move.from_square
            assert recovered.to_square == move.to_square

    def test_promotion_auto_queen(self):
        # Position where white pawn is about to promote
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        for move in board.legal_moves:
            if move.promotion:
                idx = move_to_index(move)
                recovered = index_to_move(idx, board)
                # auto-promote to queen
                assert recovered.promotion == chess.QUEEN

    def test_ep_square_encoding(self):
        # After 1. e4 e5 2. e5... actually after e4 d5 the ep square is set
        board = chess.Board()
        board.push_uci("e2e4")  # white pawn double push → ep square = e3
        _, et = encode_board(board)
        # ep square should be encoded, not 10 (none)
        assert et[5].item() != 10

    def test_side_to_move_encoding(self):
        board = chess.Board()
        _, et_white = encode_board(board)
        board.push(list(board.legal_moves)[0])
        _, et_black = encode_board(board)
        assert et_white[0].item() == 0   # white to move
        assert et_black[0].item() == 1   # black to move


# ══════════════════════════════════════════════════════════════════════════════
# genetic.py
# ══════════════════════════════════════════════════════════════════════════════

class TestGenetic:
    def test_first_generation_size(self):
        gen = first_generation(SMALL_KWARGS)
        assert len(gen) == POPULATION_SIZE

    def test_first_generation_all_models(self):
        gen = first_generation(SMALL_KWARGS)
        for m in gen:
            assert isinstance(m, ChessTransformer)

    def test_random_model_different_weights(self):
        m1 = random_model(SMALL_KWARGS)
        m2 = random_model(SMALL_KWARGS)
        # With overwhelming probability two random models differ
        p1 = list(m1.parameters())[0]
        p2 = list(m2.parameters())[0]
        assert not torch.allclose(p1, p2)

    def test_next_generation_size(self):
        models = first_generation(SMALL_KWARGS)
        n = len(models)
        # Fake tournament results
        scores = [(n - i) ** 2 for i in range(n)]
        rankings = list(range(n))
        new_gen = next_generation(models, scores, rankings, SMALL_KWARGS)
        assert len(new_gen) == POPULATION_SIZE

    def test_next_generation_types(self):
        models = first_generation(SMALL_KWARGS)
        n = len(models)
        scores = [(n - i) ** 2 for i in range(n)]
        rankings = list(range(n))
        new_gen = next_generation(models, scores, rankings, SMALL_KWARGS)
        for m in new_gen:
            assert isinstance(m, ChessTransformer)

    def test_elites_preserved(self):
        """The first 4 models in next_gen should be exact copies of the top-4."""
        models = first_generation(SMALL_KWARGS)
        n = len(models)
        scores = [(n - i) ** 2 for i in range(n)]
        rankings = list(range(n))  # model 0 is best
        new_gen = next_generation(models, scores, rankings, SMALL_KWARGS)
        for rank in range(4):
            orig = models[rankings[rank]]
            elite = new_gen[rank]
            for (name, p_orig), (_, p_elite) in zip(
                orig.named_parameters(), elite.named_parameters()
            ):
                assert torch.allclose(p_orig, p_elite), (
                    f"Elite {rank} parameter '{name}' not preserved"
                )

    def test_crossover_model_shape(self):
        parents = [random_model(SMALL_KWARGS) for _ in range(4)]
        scores = [4, 3, 2, 1]
        child = crossover_model(parents, scores, SMALL_KWARGS)
        assert isinstance(child, ChessTransformer)
        assert child.d_model == SMALL_KWARGS["d_model"]


# ══════════════════════════════════════════════════════════════════════════════
# checkpoint.py
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpoint:
    def test_save_and_load_roundtrip(self):
        models = [random_model(SMALL_KWARGS) for _ in range(3)]
        scores = [9, 4, 1]
        rankings = [0, 1, 2]
        wins = [2, 1, 0]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                tmpdir, generation=0, models=models,
                model_kwargs=SMALL_KWARGS,
                scores=scores, rankings=rankings, wins=wins,
            )
            gen, loaded, kwargs, s, r, w = load_checkpoint(
                Path(tmpdir) / "generation_0000.pt"
            )

        assert gen == 0
        assert len(loaded) == 3
        assert kwargs == SMALL_KWARGS
        assert s == scores
        assert r == rankings
        assert w == wins

    def test_weights_preserved_after_roundtrip(self):
        models = [random_model(SMALL_KWARGS)]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir, generation=5, models=models,
                            model_kwargs=SMALL_KWARGS)
            _, loaded, _, _, _, _ = load_checkpoint(
                Path(tmpdir) / "generation_0005.pt"
            )

        for (_, p_orig), (_, p_loaded) in zip(
            models[0].named_parameters(), loaded[0].named_parameters()
        ):
            assert torch.allclose(p_orig, p_loaded)

    def test_latest_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models = [random_model(SMALL_KWARGS)]
            save_checkpoint(tmpdir, generation=0, models=models,
                            model_kwargs=SMALL_KWARGS)
            latest = latest_checkpoint_path(tmpdir)
            assert latest is not None
            assert latest.exists()

    def test_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for g in range(3):
                models = [random_model(SMALL_KWARGS)]
                save_checkpoint(tmpdir, generation=g, models=models,
                                model_kwargs=SMALL_KWARGS)
            latest = latest_checkpoint_path(tmpdir)
            assert latest is not None
            gen, _, _, _, _, _ = load_checkpoint(latest)
            assert gen == 2


# ══════════════════════════════════════════════════════════════════════════════
# tournament.py  (lightweight integration – uses tiny models for speed)
# ══════════════════════════════════════════════════════════════════════════════

class TestTournament:
    def test_play_game_returns_valid_result(self):
        a = random_model(SMALL_KWARGS)
        b = random_model(SMALL_KWARGS)
        result = play_game(a, b)
        assert result in (0.0, 0.5, 1.0)

    def test_play_match_wins_non_negative(self):
        a = random_model(SMALL_KWARGS)
        b = random_model(SMALL_KWARGS)
        wa, wb = play_match(a, b)
        assert wa >= 0 and wb >= 0

    def test_play_match_max_3_decisive(self):
        a = random_model(SMALL_KWARGS)
        b = random_model(SMALL_KWARGS)
        wa, wb = play_match(a, b)
        assert wa + wb <= 3

    def test_run_tournament_scores_shape(self):
        models = [random_model(SMALL_KWARGS) for _ in range(4)]
        scores, rankings, wins = run_tournament(models)
        assert len(scores) == 4
        assert len(rankings) == 4
        assert len(wins) == 4

    def test_run_tournament_score_values(self):
        """Scores must be squares of ranks (n² down to 1²)."""
        n = 4
        models = [random_model(SMALL_KWARGS) for _ in range(n)]
        scores, rankings, wins = run_tournament(models)
        # Collect all score values and check they form {1,4,9,16}
        assert set(scores) == {(n - i) ** 2 for i in range(n)}

    def test_run_tournament_rankings_permutation(self):
        """Rankings must be a permutation of model indices."""
        n = 4
        models = [random_model(SMALL_KWARGS) for _ in range(n)]
        _, rankings, _ = run_tournament(models)
        assert sorted(rankings) == list(range(n))
