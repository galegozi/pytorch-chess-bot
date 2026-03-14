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
Moves are encoded across four 4096-entry *planes* (total 16 384 indices):

  plane 0  [    0 – 4095] : all regular moves + queen promotions
  plane 1  [ 4096 – 8191] : knight underpromotions
  plane 2  [ 8192 –12287] : bishop underpromotions
  plane 3  [12288 –16383] : rook underpromotions

Within each plane the index is ``from_square * 64 + to_square``.  A queen
promotion (or any non-promoting move) lives in plane 0; a knight/bishop/rook
promotion lives in the corresponding plane while sharing the same from/to
squares.  This allows the model to express a preference for underpromotions.
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

NUM_MOVE_INDICES = 64 * 64 * 4  # 16384  (4 planes: queen, knight, bishop, rook)

# Underpromotion pieces in plane order (plane 1, 2, 3)
_UNDERPROMO_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


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
    """Return a boolean tensor of shape ``(16384,)`` marking legal moves.

    Queen promotions occupy plane 0 (indices 0–4095); knight, bishop, and rook
    underpromotions occupy planes 1–3 respectively.
    """
    mask = torch.zeros(NUM_MOVE_INDICES, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask


def move_to_index(move: chess.Move) -> int:
    """Encode a ``chess.Move`` as an integer in ``[0, 16383]``.

    Queen promotions (and all non-promoting moves) map to plane 0
    (``from_square * 64 + to_square``).  Knight, bishop, and rook
    underpromotions map to planes 1, 2, and 3 respectively.
    """
    base = move.from_square * 64 + move.to_square
    if move.promotion is not None and move.promotion != chess.QUEEN:
        plane = _UNDERPROMO_PIECES.index(move.promotion) + 1
        return 4096 * plane + base
    return base


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Decode a move index to a ``chess.Move``.

    Plane 0 (indices 0–4095): queen promotion when the pawn reaches the back
    rank, otherwise a regular move.  Planes 1–3 (4096–16383): knight, bishop,
    and rook underpromotions respectively.
    """
    plane  = idx // 4096
    base   = idx % 4096
    from_sq = base // 64
    to_sq   = base % 64
    piece = board.piece_at(from_sq)
    is_promo = (
        piece is not None
        and piece.piece_type == chess.PAWN
        and (
            (piece.color == chess.WHITE and chess.square_rank(to_sq) == 7)
            or (piece.color == chess.BLACK and chess.square_rank(to_sq) == 0)
        )
    )
    if is_promo:
        promo_piece = _UNDERPROMO_PIECES[plane - 1] if plane > 0 else chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promo_piece)
    return chess.Move(from_sq, to_sq)
