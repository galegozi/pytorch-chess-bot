"""
Chess Transformer Model

A resizable PyTorch transformer that encodes a chess position as a token
sequence and predicts a probability distribution over all move indices.
Moves are encoded across four 4096-entry planes (16 384 total): plane 0 for
regular moves and queen promotions, planes 1–3 for knight, bishop, and rook
underpromotions respectively.
"""

import torch
import torch.nn as nn


class ChessTransformer(nn.Module):
    """Transformer model that maps a chess position to move logits.

    Board state is encoded as 64 piece tokens (one per square in SQUARES order)
    plus 6 metadata tokens: side-to-move, the four castling rights, and the
    en-passant square.  A learned positional embedding is added to each token
    before feeding into the encoder stack.

    The policy head mean-pools the encoder output over the sequence dimension
    and projects it to ``num_moves`` logits.  The default of 16 384 covers four
    4096-entry planes: plane 0 for regular moves and queen promotions, planes
    1–3 for knight, bishop, and rook underpromotions respectively.

    Args:
        d_model:              Internal embedding dimension.
        nhead:                Number of attention heads (must divide ``d_model``).
        num_encoder_layers:   Number of transformer encoder layers.
        dim_feedforward:      Hidden size of each feed-forward sublayer.
        dropout:              Dropout probability used inside the encoder.
        num_piece_types:      Vocabulary size for piece tokens (13: empty + 6 white + 6 black).
        num_extra_token_vals: Vocabulary size for metadata tokens (128 is safe).
        num_extra_tokens:     Number of metadata tokens appended to the board sequence.
        num_moves:            Output vocabulary size (16 384 = 4×64×64 by default).
    """

    # Canonical constructor kwargs used for checkpointing
    CONFIG_KEYS = (
        "d_model",
        "nhead",
        "num_encoder_layers",
        "dim_feedforward",
        "dropout",
        "num_piece_types",
        "num_extra_token_vals",
        "num_extra_tokens",
        "num_moves",
    )

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_piece_types: int = 13,
        num_extra_token_vals: int = 128,
        num_extra_tokens: int = 6,
        num_moves: int = 16384,
    ) -> None:
        super().__init__()

        # Store all config values for checkpoint / resume
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_piece_types = num_piece_types
        self.num_extra_token_vals = num_extra_token_vals
        self.num_extra_tokens = num_extra_tokens
        self.num_moves = num_moves

        self.seq_len = 64 + num_extra_tokens

        # ── Embeddings ────────────────────────────────────────────────────────
        self.piece_embedding = nn.Embedding(num_piece_types, d_model)
        self.extra_embedding = nn.Embedding(num_extra_token_vals, d_model)
        self.pos_embedding = nn.Embedding(self.seq_len, d_model)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # ── Policy head (mean-pool → linear) ──────────────────────────────────
        self.policy_head = nn.Linear(d_model, num_moves)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        board_tokens: torch.Tensor,
        extra_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute move logits for a batch of positions.

        Args:
            board_tokens:  ``(B, 64)`` int64 – piece-type index per square.
            extra_tokens:  ``(B, num_extra_tokens)`` int64 – metadata tokens.

        Returns:
            logits: ``(B, num_moves)`` float – unnormalised move scores.
        """
        x_board = self.piece_embedding(board_tokens)   # (B, 64, d_model)
        x_extra = self.extra_embedding(extra_tokens)   # (B, E,  d_model)
        x = torch.cat([x_board, x_extra], dim=1)       # (B, seq_len, d_model)

        positions = torch.arange(self.seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)          # add positional info

        x = self.transformer(x)                        # (B, seq_len, d_model)
        x = x.mean(dim=1)                              # (B, d_model)
        return self.policy_head(x)                     # (B, num_moves)

    # ── Inference helper ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_move(
        self,
        board_tokens: torch.Tensor,
        extra_tokens: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Greedily select the highest-scored *legal* move.

        Args:
            board_tokens:  ``(1, 64)`` int64.
            extra_tokens:  ``(1, num_extra_tokens)`` int64.
            legal_mask:    ``(1, num_moves)`` bool – True for legal moves.

        Returns:
            move_idx: ``(1,)`` int64 – selected move index.
        """
        self.eval()
        logits = self.forward(board_tokens, extra_tokens)
        logits = logits.masked_fill(~legal_mask, float("-inf"))
        return logits.argmax(dim=-1)

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return constructor kwargs needed to recreate this model."""
        return {k: getattr(self, k) for k in self.CONFIG_KEYS}


def build_model(config: dict) -> "ChessTransformer":
    """Build a ``ChessTransformer`` from a config dict."""
    return ChessTransformer(**config)
