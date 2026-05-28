# Chess Cheat Detection

Engine-assisted move detection in a controlled Stockfish self-play environment. Academic project for an AI course.

## Experimental thesis

Real cheaters use engines; the fundamental problem is that high-quality public labels are not available. Rather than pretending otherwise, we build a synthetic environment with perfect ground truth:

- **"Honest" class**: Stockfish at a low Skill Level plays against another Stockfish at the same Skill Level.
- **"Cheat" class**: same configuration, but one side replaces a percentage `P` of its own moves with the top move from a full-strength Stockfish.

We systematically vary:
- **Skill Level** of the suspect player (and adversary): 0, 3, 6, 9, 12, 15, 18 — covers ~1300 to ~2500 Elo
- **Cheat percentage**: 0% (honest), 10%, 25%, 50%, 100%

Skill Level is converted to an estimated rating (numeric) and an interval bucket, based on values published by the Stockfish community. These mappings are approximations — they vary with time control and opponent — and that imprecision is discussed in the notebook.

| Skill | Rating | Bucket |
|---|---|---|
| 0  | 1320 | 1250-1400 |
| 3  | 1500 | 1400-1600 |
| 6  | 1700 | 1600-1800 |
| 9  | 1900 | 1800-2000 |
| 12 | 2100 | 2000-2200 |
| 15 | 2300 | 2200-2400 |
| 18 | 2500 | 2400-2600 |

**Balancing**: to guarantee 50/50 honest/cheat, we generate 4× more games in each honest cell (since there are 4 cheat levels for every 1 honest level per skill). With `--games-per-cell 200`, this yields 7×800=5600 honest and 7×4×200=5600 cheat = **11200 total games**.

This approach is honestly limited — the classifier learns to distinguish "pure weak engine" from "weak engine with injections", not "honest human" from "human with cheat". A weak engine's move distribution is *not* the same as a human's, and so the model does not directly transfer to real detection. This limitation is discussed in the notebook and in the report, and is considered part of the work's contribution.

## Structure

```
chess-cheat-detection/
├── README.md
├── requirements.txt
├── generate_games.py    # parallel self-play generator
├── notebook.ipynb       # analysis, training, evaluation 
├── data/                # games.csv 
└── results/             # plots and metrics 
```

## Setup

Prerequisites:
- Python 3.11+
- Stockfish binary accessible (default: `C:\stockfish\stockfish.exe`, override via env var `STOCKFISH_PATH`)

```powershell
# inside the project venv
pip install -r requirements.txt
```

## Game generation

```powershell
# smoke test (fast, ~2min, generates ~112 games)
python generate_games.py --games-per-cell 2

# full run (~8h on a Ryzen 5 7600 with 12 workers, generates 11200 games)
python generate_games.py --games-per-cell 200
```

Output: `data/games.csv` with one row per game + aggregated features + metadata (`skill_level`, `estimated_rating`, `rating_bucket`, `cheat_pct`, `is_cheat`).

## Training

See `notebook.ipynb`.
