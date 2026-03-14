"""
lichess_bot.py – Lichess bot interface for the chess transformer.

The bot connects to Lichess using the official Bot API (via the ``berserk``
client library), listens for incoming game challenges, accepts them, and plays
moves using the selected ChessTransformer model.

Prerequisites
-------------
1. A Lichess account that has been *upgraded to a bot account*:
      https://lichess.org/api#operation/botAccountUpgrade
2. A personal API token with the ``bot:play`` scope:
      https://lichess.org/account/oauth/token/create
3. Install dependencies:
      pip install berserk chess torch

Usage
-----
Set your token as an environment variable (recommended) or pass via --token:

    export LICHESS_TOKEN="<your_lichess_api_token>"
    python lichess_bot.py

    # Or point to a specific checkpoint
    python lichess_bot.py --model checkpoints/generation_0010.pt --model-index 0

Full option list:
    python lichess_bot.py --help

How games are played
--------------------
* The bot accepts *all* incoming challenges (time-control-agnostic).
* On each move it encodes the current board, runs a forward pass through the
  transformer, and selects the highest-scoring legal move (greedy).
* If no legal move is predicted (shouldn't happen), it falls back to a random
  legal move.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import threading

import chess
import berserk

from chess_utils import encode_board, legal_moves_mask, index_to_move
from checkpoint import load_checkpoint, latest_checkpoint_path
from model import ChessTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lichess bot interface for the chess transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Lichess API token with bot:play scope.  "
            "Defaults to the LICHESS_TOKEN environment variable."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint .pt file.  "
            "Defaults to the latest checkpoint in --checkpoint-dir."
        ),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to search for checkpoints when --model is not given.",
    )
    p.add_argument(
        "--model-index",
        type=int,
        default=0,
        help=(
            "Index into the checkpoint's population list to use as the bot "
            "model.  0 = first model stored (use after sorting by ranking)."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (cpu / cuda / mps).",
    )
    return p


# ── Bot logic ─────────────────────────────────────────────────────────────────

class ChessBot:
    """Wraps a ``ChessTransformer`` and interacts with the Lichess Bot API."""

    def __init__(self, client: berserk.Client, model: ChessTransformer, device: str) -> None:
        self.client = client
        self.model = model
        self.device = device

    # ── Move selection ────────────────────────────────────────────────────────

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Select the best legal move for the current position."""
        board_tokens, extra_tokens = encode_board(board)
        mask = legal_moves_mask(board)

        bt = board_tokens.unsqueeze(0).to(self.device)
        et = extra_tokens.unsqueeze(0).to(self.device)
        lm = mask.unsqueeze(0).to(self.device)

        move_idx = self.model.select_move(bt, et, lm).item()
        move = index_to_move(move_idx, board)

        if move in board.legal_moves:
            return move

        # Fallback: random legal move
        logger.warning("Model selected an illegal move; falling back to random.")
        return random.choice(list(board.legal_moves))

    # ── Game handler ──────────────────────────────────────────────────────────

    def handle_game(self, game_id: str) -> None:
        """Stream and play a single game."""
        logger.info("Game started: %s", game_id)
        board = chess.Board()
        bot_color: bool | None = None  # True = white, False = black

        try:
            for event in self.client.bots.stream_game_state(game_id):
                event_type = event.get("type")

                # ── gameFull: initial state sent at the start ─────────────────
                if event_type == "gameFull":
                    white_id = event["white"].get("id", "")
                    me = self.client.account.get()["id"]
                    bot_color = white_id == me

                    # Replay all moves played before we started listening
                    moves_str = event.get("state", {}).get("moves", "")
                    board = chess.Board()
                    for uci in moves_str.split():
                        board.push_uci(uci)

                # ── gameState: move update ────────────────────────────────────
                elif event_type == "gameState":
                    moves_str = event.get("moves", "")
                    board = chess.Board()
                    for uci in moves_str.split():
                        board.push_uci(uci)

                    status = event.get("status", "started")
                    if status not in ("started", "created"):
                        logger.info("Game %s ended with status: %s", game_id, status)
                        break

                # After both events, decide whether it's our turn
                if bot_color is None:
                    continue
                if board.is_game_over():
                    break
                our_turn = (board.turn == chess.WHITE) == bot_color
                if not our_turn:
                    continue

                move = self.choose_move(board)
                self.client.bots.make_move(game_id, move.uci())
                logger.info("Game %s: played %s", game_id, move.uci())

        except Exception as exc:
            logger.error("Error in game %s: %s", game_id, exc)

    # ── Event loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main event loop: accept challenges and spawn game threads."""
        logger.info("Bot is running and listening for events …")
        for event in self.client.bots.stream_incoming_events():
            event_type = event.get("type")
            logger.debug("Event: %s", event_type)

            if event_type == "challenge":
                challenge_id = event["challenge"]["id"]
                logger.info("Accepting challenge %s", challenge_id)
                try:
                    self.client.bots.accept_challenge(challenge_id)
                except Exception as exc:
                    logger.warning("Could not accept challenge %s: %s", challenge_id, exc)

            elif event_type == "gameStart":
                game_id = event["game"]["gameId"]
                thread = threading.Thread(
                    target=self.handle_game, args=(game_id,), daemon=True
                )
                thread.start()


# ── Entry point ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # ── API token ────────────────────────────────────────────────────────────
    token = args.token or os.environ.get("LICHESS_TOKEN")
    if not token:
        print(
            "Error: Lichess API token not provided.  "
            "Use --token or set the LICHESS_TOKEN environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────────
    model_path = args.model
    if model_path is None:
        model_path = latest_checkpoint_path(args.checkpoint_dir)
        if model_path is None:
            print(
                f"Error: no checkpoint found in '{args.checkpoint_dir}'.  "
                "Train a model first with train.py, or use --model.",
                file=sys.stderr,
            )
            sys.exit(1)

    logger.info("Loading model from %s (index %d)", model_path, args.model_index)
    _gen, models, _kwargs, _scores, rankings, _wins = load_checkpoint(
        model_path, device=args.device
    )

    # If rankings are available, pick the best model; otherwise use model_index
    if rankings is not None and args.model_index == 0:
        idx = rankings[0]
        logger.info("Using best model from checkpoint (index %d by ranking)", idx)
    else:
        idx = args.model_index
        logger.info("Using model at population index %d", idx)

    model = models[idx]
    model.eval()

    # ── Lichess client ───────────────────────────────────────────────────────
    session = berserk.TokenSession(token)
    client = berserk.Client(session)

    try:
        me = client.account.get()
        logger.info("Logged in as: %s (title: %s)", me["id"], me.get("title", "—"))
    except Exception as exc:
        logger.error("Failed to authenticate with Lichess: %s", exc)
        sys.exit(1)

    # ── Start bot ────────────────────────────────────────────────────────────
    bot = ChessBot(client, model, args.device)
    bot.run()


if __name__ == "__main__":
    main()
