"""Search for better evaluate() heuristic weights via self-play.

Weights are (w_empty, w_merge, w_mono, corner_bonus, w_smooth) threaded
through ai2048.fastcore.best_move_fast(..., weights=...) as plain runtime
arguments (not numba globals -- numba bakes in global constants at
compile time, so mutating fastcore.W_EMPTY etc. after import would
silently not change already-jitted behavior).

Usage:
    python tune.py            # run the search, print the winner
"""
import random
import time

from ai2048.game import Game
from ai2048.fastcore import best_move_fast, DEFAULT_WEIGHTS

# Fast settings for the search phase itself (cheap per game). The final
# winner should be re-verified at real shipped settings separately.
TUNE_TIME = 0.08
TUNE_MIN_DEPTH = 2
TUNE_MAX_DEPTH = 6
TUNE_SEEDS = (1, 2, 3)


def play_game(weights, seed, time_limit=TUNE_TIME, min_depth=TUNE_MIN_DEPTH, max_depth=TUNE_MAX_DEPTH, move_cap=4000):
    rng = random.Random(seed)
    g = Game(rng)
    moves = 0
    while not g.is_game_over() and moves < move_cap:
        mv = best_move_fast(g.board, time_limit=time_limit, min_depth=min_depth, max_depth=max_depth, weights=weights)
        if mv is None:
            break
        g.step(mv)
        moves += 1
    return g.score, g.max_tile(), moves


def objective(weights, seeds=TUNE_SEEDS, **kw):
    scores = []
    tiles = []
    for s in seeds:
        sc, mt, _ = play_game(weights, s, **kw)
        scores.append(sc)
        tiles.append(mt)
    avg_score = sum(scores) / len(scores)
    frac_2048 = sum(1 for t in tiles if t >= 2048) / len(tiles)
    frac_4096 = sum(1 for t in tiles if t >= 4096) / len(tiles)
    # weight toward reliably reaching bigger tiles, not just raw score
    obj = avg_score + 8000.0 * frac_2048 + 16000.0 * frac_4096
    return obj, avg_score, tiles


def coordinate_ascent():
    names = ("w_empty", "w_merge", "w_mono", "corner_bonus", "w_smooth")
    current = list(DEFAULT_WEIGHTS)
    best_obj, best_avg, best_tiles = objective(tuple(current))
    print(f"baseline weights={tuple(current)} obj={best_obj:.0f} avg_score={best_avg:.0f} tiles={best_tiles}")

    # try adding a small smoothness term once as a structural A/B, independent of the sweep below
    for w_smooth_try in (10.0, 30.0):
        cand = list(current)
        cand[4] = w_smooth_try
        obj, avg, tiles = objective(tuple(cand))
        print(f"  smoothness w={w_smooth_try}: obj={obj:.0f} avg_score={avg:.0f} tiles={tiles}")
        if obj > best_obj:
            best_obj, best_avg, best_tiles, current = obj, avg, tiles, cand

    improved = True
    rounds = 0
    while improved and rounds < 2:
        improved = False
        rounds += 1
        for i, name in enumerate(names):
            if name == "w_smooth":
                continue  # already explored above; keep whatever won
            base_val = current[i]
            for factor in (0.6, 1.4):
                cand = list(current)
                cand[i] = base_val * factor if base_val != 0 else (10.0 if factor > 1 else -10.0)
                obj, avg, tiles = objective(tuple(cand))
                print(f"  round{rounds} {name}*{factor}: weights={tuple(cand)} obj={obj:.0f} avg_score={avg:.0f} tiles={tiles}")
                if obj > best_obj:
                    best_obj, best_avg, best_tiles = obj, avg, tiles
                    current = cand
                    improved = True
                    print(f"    -> new best obj={best_obj:.0f}")

    return tuple(current), best_obj, best_avg, best_tiles


# Real shipped settings, used only for the final verification pass below
# -- NOT for the search itself (too expensive per game, see TUNE_* above).
VERIFY_TIME = 0.15
VERIFY_MIN_DEPTH = 3
VERIFY_MAX_DEPTH = 7
VERIFY_SEEDS = (1, 2, 3, 4, 5)


def verify_at_real_settings(weights, label):
    scores, tiles = [], []
    for s in VERIFY_SEEDS:
        sc, mt, _ = play_game(weights, s, time_limit=VERIFY_TIME, min_depth=VERIFY_MIN_DEPTH, max_depth=VERIFY_MAX_DEPTH)
        scores.append(sc)
        tiles.append(mt)
        print(f"  [{label}] seed={s} score={sc} max_tile={mt}")
    avg = sum(scores) / len(scores)
    f2048 = sum(1 for t in tiles if t >= 2048) / len(tiles)
    f4096 = sum(1 for t in tiles if t >= 4096) / len(tiles)
    print(f"  [{label}] avg_score={avg:.0f} best={max(scores)} >=2048:{f2048*100:.0f}% >=4096:{f4096*100:.0f}%")
    return avg, f2048, f4096


if __name__ == "__main__":
    t0 = time.perf_counter()
    best_weights, best_obj, best_avg, best_tiles = coordinate_ascent()
    print(f"\nSearch done in {time.perf_counter() - t0:.1f}s (fast tuning settings, noisy -- only {len(TUNE_SEEDS)} seeds)")
    print(f"Candidate weights: {best_weights}  (baseline: {DEFAULT_WEIGHTS})")

    if best_weights == tuple(DEFAULT_WEIGHTS):
        print("\nSearch did not find anything better than baseline. No change.")
    else:
        print("\n=== Re-verifying candidate vs. baseline at REAL shipped settings ===")
        print("NOTE: weights tuned at shallow/fast search settings do not reliably")
        print("transfer to deeper search -- always re-check here before trusting them.")
        print("baseline:")
        base_avg, base_f2048, base_f4096 = verify_at_real_settings(DEFAULT_WEIGHTS, "baseline")
        print("candidate:")
        cand_avg, cand_f2048, cand_f4096 = verify_at_real_settings(best_weights, "candidate")

        if cand_avg > base_avg and cand_f2048 >= base_f2048 * 0.9:
            print(f"\nCandidate wins at real settings (avg {cand_avg:.0f} vs {base_avg:.0f}).")
            print("-> Update W_EMPTY/W_MERGE/W_MONO/CORNER_BONUS/W_SMOOTH in ai2048/fastcore.py to this tuple.")
        else:
            print(f"\nCandidate does NOT clearly beat baseline at real settings "
                  f"(avg {cand_avg:.0f} vs {base_avg:.0f}, >=2048 {cand_f2048*100:.0f}% vs {base_f2048*100:.0f}%).")
            print("Fast-setting tuning overfit to shallow search; keeping baseline weights in fastcore.py.")
