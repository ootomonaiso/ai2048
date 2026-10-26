"""Benchmark the Expectimax AI over N headless games."""
import argparse
import random
import time
from collections import Counter

from ai2048.game import Game
from ai2048.fastcore import best_move_fast


def run_one(seed, time_limit, min_depth, max_depth):
    rng = random.Random(seed)
    game = Game(rng)
    moves = 0
    while not game.is_game_over():
        move_idx = best_move_fast(game.board, time_limit=time_limit, min_depth=min_depth, max_depth=max_depth)
        if move_idx is None:
            break
        game.step(move_idx)
        moves += 1
    return game.score, game.max_tile(), moves


def main():
    parser = argparse.ArgumentParser(description="Benchmark the Expectimax AI over N games")
    parser.add_argument("-n", "--games", type=int, default=10)
    parser.add_argument("--time", type=float, default=0.3)
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tile_counter = Counter()
    scores = []
    t0 = time.perf_counter()
    for i in range(args.games):
        score, max_tile, moves = run_one(args.seed + i, args.time, args.min_depth, args.max_depth)
        scores.append(score)
        tile_counter[max_tile] += 1
        print(f"game {i + 1}/{args.games}: score={score} max_tile={max_tile} moves={moves}")
    elapsed = time.perf_counter() - t0

    print("\n=== Summary ===")
    print(f"games: {args.games}  elapsed: {elapsed:.1f}s")
    print(f"avg score: {sum(scores) / len(scores):.0f}")
    print(f"best score: {max(scores)}")
    for tile in sorted(tile_counter, reverse=True):
        reached = sum(c for t, c in tile_counter.items() if t >= tile)
        pct = 100.0 * reached / args.games
        print(f"  reached >= {tile:>6}: {reached}/{args.games} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
