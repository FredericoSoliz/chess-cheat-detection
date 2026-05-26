"""Generate Stockfish self-play games with controlled cheat injection.

For each (skill_level, cheat_pct) cell, plays N games where one side (the
"suspect") plays at the given Skill Level but, with probability cheat_pct/100,
replaces its own move with the top move from a full-strength Stockfish.

Per-game features are extracted from the suspect's moves and written to CSV.
"""
import argparse
import atexit
import csv
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.engine
from tqdm import tqdm

DEFAULT_STOCKFISH = r"C:\stockfish\stockfish.exe"
STRONG_DEPTH = 14
WEAK_MOVETIME_S = 0.03
MAX_PLIES = 200
CP_LOSS_CAP = 1000  # cap per-move centipawn loss; standard Lichess-style

SKILL_LEVELS = [0, 3, 6, 9, 12, 15, 18]
CHEAT_PCTS = [0, 10, 25, 50, 100]

SKILL_TO_RATING = {
    0: 1320,
    3: 1500,
    6: 1700,
    9: 1900,
    12: 2100,
    15: 2300,
    18: 2500,
}
SKILL_TO_BUCKET = {
    0: "1250-1400",
    3: "1400-1600",
    6: "1600-1800",
    9: "1800-2000",
    12: "2000-2200",
    15: "2200-2400",
    18: "2400-2600",
}

HONEST_MULTIPLIER = 4  # number of cheat cells per skill; honest cell scales to keep 50/50


@dataclass
class GameFeatures:
    game_id: int
    skill_level: int
    estimated_rating: int
    rating_bucket: str
    cheat_pct: int
    is_cheat: int
    suspect_color: str
    n_moves: int
    result: str
    acpl: float
    acpl_opening: float
    acpl_middlegame: float
    acpl_endgame: float
    top1_match_rate: float
    top3_match_rate: float
    cp_loss_std: float
    cp_loss_median: float
    only_move_match_rate: float
    blunder_rate: float
    mistake_rate: float


def score_cp(score_obj, pov_color):
    s = score_obj.pov(pov_color)
    if s.is_mate():
        return 10000 if s.mate() > 0 else -10000
    return s.score()


def classify_phase(board: chess.Board) -> str:
    if board.ply() < 20:
        return "opening"
    non_pk = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None and p.piece_type not in (chess.KING, chess.PAWN):
            non_pk += 1
    return "endgame" if non_pk <= 6 else "middlegame"


_W = {}  # per-worker persistent engines


def _worker_cleanup():
    for k in ("strong", "weak_a", "weak_b"):
        eng = _W.get(k)
        if eng is not None:
            try:
                eng.quit()
            except Exception:
                pass
            _W[k] = None


def _worker_init(stockfish_path):
    _W["path"] = stockfish_path
    _W["strong"] = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _W["weak_a"] = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _W["weak_b"] = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    atexit.register(_worker_cleanup)


def _ensure_engine(key):
    eng = _W.get(key)
    if eng is None:
        _W[key] = chess.engine.SimpleEngine.popen_uci(_W["path"])
        return _W[key]
    return eng


def _reset_engine(key):
    try:
        _W[key].quit()
    except Exception:
        pass
    _W[key] = chess.engine.SimpleEngine.popen_uci(_W["path"])


def play_one_game(args):
    game_id, skill_level, cheat_pct, seed = args
    rng = random.Random(seed)
    suspect_color = chess.WHITE if rng.random() < 0.5 else chess.BLACK
    board = chess.Board()
    per_move = []

    try:
        strong = _ensure_engine("strong")
        weak_suspect = _ensure_engine("weak_a")
        weak_opp = _ensure_engine("weak_b")
        weak_suspect.configure({"Skill Level": skill_level})
        weak_opp.configure({"Skill Level": skill_level})

        while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
            if board.turn == suspect_color:
                info_list = strong.analyse(
                    board,
                    chess.engine.Limit(depth=STRONG_DEPTH),
                    multipv=3,
                )
                top_moves = [pv["pv"][0] for pv in info_list if pv.get("pv")]
                if not top_moves:
                    break
                score_before = score_cp(info_list[0]["score"], suspect_color)

                if len(info_list) >= 2:
                    second_score = score_cp(info_list[1]["score"], suspect_color)
                    is_only_move = (score_before - second_score) > 200
                else:
                    is_only_move = True

                if rng.random() * 100 < cheat_pct:
                    chosen = top_moves[0]
                else:
                    res = weak_suspect.play(board, chess.engine.Limit(time=WEAK_MOVETIME_S))
                    chosen = res.move
                    if chosen is None:
                        break

                phase = classify_phase(board)
                top1 = 1 if chosen == top_moves[0] else 0
                top3 = 1 if chosen in top_moves[:3] else 0

                board.push(chosen)

                if board.is_game_over(claim_draw=True):
                    score_after = score_before
                else:
                    info_after = strong.analyse(board, chess.engine.Limit(depth=STRONG_DEPTH))
                    score_after = score_cp(info_after["score"], suspect_color)
                cp_loss = min(CP_LOSS_CAP, max(0, score_before - score_after))

                per_move.append({
                    "cp_loss": cp_loss,
                    "top1": top1,
                    "top3": top3,
                    "only_move_match": top1 if is_only_move else None,
                    "phase": phase,
                })
            else:
                res = weak_opp.play(board, chess.engine.Limit(time=WEAK_MOVETIME_S))
                if res.move is None:
                    break
                board.push(res.move)
    except chess.engine.EngineError as e:
        sys.stderr.write(f"[game {game_id}] engine error, respawning: {e}\n")
        for k in ("strong", "weak_a", "weak_b"):
            _reset_engine(k)
        return None
    except Exception as e:
        sys.stderr.write(f"[game {game_id}] error: {e}\n")
        return None

    if len(per_move) < 5:
        return None

    cps = [m["cp_loss"] for m in per_move]
    cps_open = [m["cp_loss"] for m in per_move if m["phase"] == "opening"]
    cps_mid = [m["cp_loss"] for m in per_move if m["phase"] == "middlegame"]
    cps_end = [m["cp_loss"] for m in per_move if m["phase"] == "endgame"]
    only_move_matches = [m["only_move_match"] for m in per_move if m["only_move_match"] is not None]

    def avg(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    def median(xs):
        if not xs:
            return 0.0
        s = sorted(xs)
        n = len(s)
        return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = avg(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    return GameFeatures(
        game_id=game_id,
        skill_level=skill_level,
        estimated_rating=SKILL_TO_RATING[skill_level],
        rating_bucket=SKILL_TO_BUCKET[skill_level],
        cheat_pct=cheat_pct,
        is_cheat=int(cheat_pct > 0),
        suspect_color="white" if suspect_color == chess.WHITE else "black",
        n_moves=len(per_move),
        result=board.result(claim_draw=True),
        acpl=avg(cps),
        acpl_opening=avg(cps_open),
        acpl_middlegame=avg(cps_mid),
        acpl_endgame=avg(cps_end),
        top1_match_rate=avg([m["top1"] for m in per_move]),
        top3_match_rate=avg([m["top3"] for m in per_move]),
        cp_loss_std=std(cps),
        cp_loss_median=median(cps),
        only_move_match_rate=avg(only_move_matches),
        blunder_rate=avg([1.0 if c > 200 else 0.0 for c in cps]),
        mistake_rate=avg([1.0 if c > 100 else 0.0 for c in cps]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-cell", type=int, default=200)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=str, default="data/games.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stockfish",
        type=str,
        default=os.environ.get("STOCKFISH_PATH", DEFAULT_STOCKFISH),
    )
    args = parser.parse_args()

    if not Path(args.stockfish).exists():
        sys.exit(
            f"Stockfish not found at {args.stockfish}. "
            f"Set STOCKFISH_PATH or pass --stockfish."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    game_id = 0
    rng = random.Random(args.seed)
    n_cheat_pcts = sum(1 for c in CHEAT_PCTS if c > 0)
    for skill in SKILL_LEVELS:
        for cheat in CHEAT_PCTS:
            n = args.games_per_cell * (n_cheat_pcts if cheat == 0 else 1)
            for _ in range(n):
                tasks.append((game_id, skill, cheat, rng.randint(0, 2**31 - 1)))
                game_id += 1

    print(f"Generating {len(tasks)} games with {args.workers} workers...")

    fieldnames = [f.name for f in GameFeatures.__dataclass_fields__.values()]
    n_written = 0
    failures = 0
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(args.stockfish,),
        ) as ex:
            futures = [ex.submit(play_one_game, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    r = fut.result()
                except Exception as e:
                    sys.stderr.write(f"worker error: {e}\n")
                    failures += 1
                    continue
                if r is None:
                    failures += 1
                    continue
                writer.writerow(asdict(r))
                n_written += 1
                if n_written % 100 == 0:
                    f.flush()
            f.flush()
            print(f"Wrote {n_written} games to {args.output} ({failures} failures)")
            print("Shutting down workers...")


if __name__ == "__main__":
    main()
