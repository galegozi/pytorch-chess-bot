"""
Chess board encoding and move utilities.

Encodes a ``chess.Board`` as two integer tensors that can be fed directly
into ``ChessTransformer.forward``:

* ``board_tokens``  – shape ``(64,)`` – piece-type index for every square in
  ``chess.SQUARES`` order (A1 = index 0, H8 = index 63).
* ``extra_tokens``  – shape ``(6,)``  – side-to-move, four castling-right
  flags, and the en-passant target square.

Token value ranges
------------------
Piece tokens (board_tokens):
  0  empty
  1  white pawn       7  black pawn
  2  white knight     8  black knight
  3  white bishop     9  black bishop
  4  white rook      10  black rook
  5  white queen     11  black queen
  6  white king      12  black king

Extra tokens (extra_tokens), shared vocabulary, values in [0, 74]:
  token[0] side-to-move   : 0 = white, 1 = black
  token[1] castle WK      : 2 = yes,   3 = no
  token[2] castle WQ      : 4 = yes,   5 = no
  token[3] castle BK      : 6 = yes,   7 = no
  token[4] castle BQ      : 8 = yes,   9 = no
  token[5] en-passant sq  : 10 = none, 11+sq for sq in [0,63]

  Note on en passant: chess rules guarantee at most ONE en passant target
  square per position (only one pawn can double-push per move).  Multiple
  opponent pawns may all be able to capture that single target square; each of
  those captures appears as a distinct move in ``board.legal_moves`` and is
  therefore independently set in the legal-move mask.  ``board.ep_square`` is
  always a single ``Optional[int]``, so a single extra token is sufficient.

Move encoding
-------------
Moves are encoded as ``from_square * 64 + to_square`` (range 0–4095).
Pawn promotions to non-queen pieces are always promoted to queen for
simplicity.
"""

from __future__ import annotations

import chess
import torch

# ── Piece vocabulary ──────────────────────────────────────────────────────────

_PIECE_MAP: dict[chess.Piece | None, int] = {
    None: 0,
    chess.Piece(chess.PAWN,   chess.WHITE): 1,
    chess.Piece(chess.KNIGHT, chess.WHITE): 2,
    chess.Piece(chess.BISHOP, chess.WHITE): 3,
    chess.Piece(chess.ROOK,   chess.WHITE): 4,
    chess.Piece(chess.QUEEN,  chess.WHITE): 5,
    chess.Piece(chess.KING,   chess.WHITE): 6,
    chess.Piece(chess.PAWN,   chess.BLACK): 7,
    chess.Piece(chess.KNIGHT, chess.BLACK): 8,
    chess.Piece(chess.BISHOP, chess.BLACK): 9,
    chess.Piece(chess.ROOK,   chess.BLACK): 10,
    chess.Piece(chess.QUEEN,  chess.BLACK): 11,
    chess.Piece(chess.KING,   chess.BLACK): 12,
}

NUM_MOVE_INDICES = 64 * 64  # 4096


# ── Public API ─────────────────────────────────────────────────────────────────

def encode_board(board: chess.Board) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a board position into token tensors.

    Returns
    -------
    board_tokens : torch.Tensor  shape ``(64,)``  dtype int64
    extra_tokens : torch.Tensor  shape ``(6,)``   dtype int64
    """
    # Board tokens – one per square
    board_tokens = torch.tensor(
        [_PIECE_MAP.get(board.piece_at(sq), 0) for sq in chess.SQUARES],
        dtype=torch.long,
    )

    # Extra tokens
    extra = [
        0 if board.turn == chess.WHITE else 1,                                 # side to move
        2 if board.has_kingside_castling_rights(chess.WHITE) else 3,           # WK castle
        4 if board.has_queenside_castling_rights(chess.WHITE) else 5,          # WQ castle
        6 if board.has_kingside_castling_rights(chess.BLACK) else 7,           # BK castle
        8 if board.has_queenside_castling_rights(chess.BLACK) else 9,          # BQ castle
        10 if board.ep_square is None else 11 + board.ep_square,               # ep square
    ]
    extra_tokens = torch.tensor(extra, dtype=torch.long)

    return board_tokens, extra_tokens


def legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """Return a boolean tensor of shape ``(4096,)`` marking legal moves.

    Only from/to pairs are considered; promotion pieces are ignored (we always
    promote to queen).
    """
    mask = torch.zeros(NUM_MOVE_INDICES, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask


def move_to_index(move: chess.Move) -> int:
    """Encode a ``chess.Move`` as an integer in ``[0, 4095]``."""
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Decode a move index to a ``chess.Move``, auto-promoting to queen.

    If the index corresponds to a pawn reaching the back rank, the promotion
    piece is set to queen.
    """
    from_sq = idx // 64
    to_sq = idx % 64
    piece = board.piece_at(from_sq)
    if (
        piece is not None
        and piece.piece_type == chess.PAWN
        and (
            (piece.color == chess.WHITE and chess.square_rank(to_sq) == 7)
            or (piece.color == chess.BLACK and chess.square_rank(to_sq) == 0)
        )
    ):
        return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    return chess.Move(from_sq, to_sq)
