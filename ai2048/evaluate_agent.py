"""Agent-agnostic strength evaluation harness.

Plays N games driven by any move-selection callable and reports the
max-tile reach-rate distribution + score stats. Used both to validate a
trained net against the hand-heuristic baseline and to log progress
during weeks-long training (call it periodically on the current weights).

A "policy" is any callable `policy(board_int) -> move_idx (0..3) or None`.
None means "no legal move / resign" and ends the game. This matches
`ai2048.fastcore.best_move_fast`'s contract, so the heuristic agent is
just `lambda b: best_move_fast(b, ...)`.
"""
import random
from collections import Counter

from . import board as B
from .game import Game

# Tiles we report reach-rates for (exponents 11..17 = 2048..131072).
REPORT_TILES = (2048, 4096, 8192, 16384, 32768, 65536, 131072)


def play_game(policy, seed, move_cap=100000):
    rng = random.Random(seed)
    g = Game(rng)
    moves = 0
    while not g.is_game_over() and moves < move_cap:
        mv = policy(g.board)
        if mv is None:
            break
        if not g.step(mv):
            # Policy returned an illegal move; fall back to any legal move
            # so a buggy/greedy policy can't silently stall the harness.
            legal = g.legal_moves()
            if not legal:
                break
            g.step(legal[0])
        moves += 1
    return g.score, g.max_tile(), moves


def evaluate(policy, games=50, seed=0, move_cap=100000, verbose=False):
    """Play `games` games and return a stats dict."""
    tile_counter = Counter()
    scores = []
    total_moves = 0
    for i in range(games):
        score, max_tile, moves = play_game(policy, seed + i, move_cap)
        scores.append(score)
        tile_counter[max_tile] += 1
        total_moves += moves
        if verbose:
            print(f"  game {i + 1}/{games}: score={score} max_tile={max_tile} moves={moves}", flush=True)

    reach = {}
    for tile in REPORT_TILES:
        reached = sum(c for t, c in tile_counter.items() if t >= tile)
        reach[tile] = reached / games if games else 0.0

    return {
        "games": games,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "best_score": max(scores) if scores else 0,
        "median_score": sorted(scores)[len(scores) // 2] if scores else 0,
        "avg_moves": total_moves / games if games else 0.0,
        "reach": reach,
        "tile_hist": dict(tile_counter),
    }


def format_stats(stats):
    lines = [
        f"games={stats['games']}  avg_score={stats['avg_score']:.0f}  "
        f"median={stats['median_score']}  best={stats['best_score']}  "
        f"avg_moves={stats['avg_moves']:.0f}"
    ]
    for tile in REPORT_TILES:
        r = stats["reach"].get(tile, 0.0)
        if r > 0.0 or tile <= 2048:
            lines.append(f"  reach >= {tile:>6}: {r * 100:5.1f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from .fastcore import best_move_fast

    parser = argparse.ArgumentParser(description="Evaluate the hand-heuristic agent")
    parser.add_argument("-n", "--games", type=int, default=20)
    parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    policy = lambda b: best_move_fast(b, time_limit=args.time, max_depth=args.max_depth)
    stats = evaluate(policy, games=args.games, seed=args.seed, verbose=True)
    print(format_stats(stats))
