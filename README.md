# pytorch-chess-bot

A **genetic-algorithm-evolved chess bot** built with PyTorch transformers.

## Overview

| Module | Responsibility |
|---|---|
| `model.py` | Resizable `ChessTransformer` – encodes a board position and predicts a move |
| `chess_utils.py` | Board → tensor encoding, legal-move masks, move index ↔ `chess.Move` |
| `tournament.py` | Round-robin best-of-3 tournament with squared-rank scoring |
| `genetic.py` | First-generation creation and next-generation breeding |
| `checkpoint.py` | Compact save / load with `latest.pt` pointer |
| `train.py` | CLI entry point for the full training loop |
| `lichess_bot.py` | Lichess Bot API interface (via `berserk`) |

---

## Quick start

```bash
pip install -r requirements.txt

# First run – 10 generations, default tiny model
python train.py --generations 10

# Resume after interruption
python train.py --generations 10 --resume

# Larger model (requires more RAM / VRAM)
python train.py --generations 10 --d-model 128 --nhead 8 --num-layers 4 --dim-feedforward 256
```

All options:

```
python train.py --help
```

---

## Model architecture

`ChessTransformer` encodes a board as a sequence of **64 piece tokens** (one per
square) plus **6 metadata tokens** (side to move, four castling-right flags,
en-passant square).  A learned positional embedding is added, the sequence is
fed through a standard PyTorch `TransformerEncoder`, and the encoder output is
mean-pooled and projected to **4 096 move logits** (64 from-squares × 64
to-squares; pawn promotions always use queen).

Default (smallest) hyperparameters:

| Parameter | Default | Effect |
|---|---|---|
| `d_model` | 64 | Embedding / attention dimension |
| `nhead` | 4 | Attention heads |
| `num_encoder_layers` | 2 | Stacked encoder layers |
| `dim_feedforward` | 128 | Feed-forward hidden size |

All parameters are configurable via CLI flags.

---

## Genetic algorithm

### Generation 0
All 20 models are created with random weights.

### Subsequent generations
After a round-robin tournament (190 matches for 20 models):

| Slot | Count | Source |
|---|---|---|
| **Elites** | 4 | Copies of the 4 best-ranked models |
| **Underdog** | 1 | Copy of the worst-ranked model |
| **Randoms** | 5 | Freshly initialised random models |
| **Mutations** | 10 | Element-wise weighted crossover over all 20 parents |

For mutation children, each individual parameter element independently draws
its value from a randomly selected parent.  The probability of choosing a
parent is proportional to its normalised squared-rank score.

### Scoring
After ranking all models by total match wins:

```
score[rank 1 / best]  = 20² = 400
score[rank 2]         = 19² = 361
…
score[rank 20 / worst] = 1² = 1
```

---

## Tournament format

* Every model plays every other model exactly once (**round-robin**).
* Each encounter is **best of 3**:
  1. Game 1 – Model A = white
  2. Game 2 – Model B = white
  3. Game 3 (only if 1–1 after two games) – colours assigned randomly
* Models never play themselves.
* Games capped at 400 half-moves to prevent infinite loops; ties/timeouts → draw.

---

## Checkpoints

Checkpoints are stored in `./checkpoints/` by default.

```
checkpoints/
  generation_0000.pt   # generation 0 (before first tournament)
  generation_0001.pt   # generation 1 (after first tournament)
  …
  latest.pt            # symlink → most recent checkpoint
```

Each file is a standard PyTorch archive containing:
* `generation` – generation index
* `model_kwargs` – architecture hyperparameters (for exact reconstruction)
* `population` – list of model `state_dict`s (compact; no optimizer state)
* `scores`, `rankings`, `wins` – tournament results

---

## Lichess bot

### Prerequisites

1. **Create a Lichess bot account** (or upgrade an existing account):
   <https://lichess.org/api#operation/botAccountUpgrade>

2. **Generate a personal API token** with `bot:play` scope:
   <https://lichess.org/account/oauth/token/create>

3. Set the token as an environment variable:

   ```bash
   export LICHESS_TOKEN="<your_lichess_api_token>"
   ```

### Running the bot

```bash
# Use the best model from the latest checkpoint
python lichess_bot.py

# Specify a checkpoint file explicitly
python lichess_bot.py --model checkpoints/generation_0010.pt

# Use a different model within the population
python lichess_bot.py --model-index 2

# All options
python lichess_bot.py --help
```

The bot:
* Accepts **all** incoming challenges automatically.
* Selects moves **greedily** (highest-scored legal move).
* Handles multiple concurrent games via threads.

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```
