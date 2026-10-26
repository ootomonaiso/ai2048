"""Watch the Expectimax AI play a single game of 2048 in the terminal."""
import argparse
import os
import random
import time

from ai2048 import board as B
from ai2048.game import Game
from ai2048.fastcore import best_move_fast


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    parser = argparse.ArgumentParser(description="Watch the Expectimax AI play 2048")
    parser.add_argument("--time", type=float, default=0.3, help="time budget per move (s)")
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=11)
    parser.add_argument("--delay", type=float, default=0.0, help="delay between moves (s)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-clear", action="store_true", help="don't clear the screen each move")
    parser.add_argument("--net", default="", help="checkpoint dir of a trained N-tuple net "
                        "(e.g. runs/ms2_5t8); if set, the net plays instead of the hand heuristic")
    parser.add_argument("--net-depth", type=int, default=1, help="fixed net depth (if --net-time 0)")
    parser.add_argument("--net-time", type=float, default=0.1, help="per-move budget (s) for adaptive "
                        "net search (default); deep in the late game. 0 = fixed --net-depth")
    args = parser.parse_args()

    policy = None
    if args.net:
        from ai2048 import netsearch as NS
        w = NS.load_weights(args.net)
        thr = NS.load_thresholds(args.net)
        if args.net_time > 0:
            policy = NS.make_policy(w, thresholds=thr, time_limit=args.net_time,
                                    max_depth=7, use_tt=True)
        else:
            policy = NS.make_policy(w, depth=args.net_depth, thresholds=thr)

    rng = random.Random(args.seed)
    game = Game(rng)

    moves_done = 0
    t_start = time.perf_counter()
    while not game.is_game_over():
        if policy is not None:
            move_idx = policy(game.board)
        else:
            move_idx = best_move_fast(
                game.board, time_limit=args.time, min_depth=args.min_depth, max_depth=args.max_depth
            )
        if move_idx is None:
            break
        game.step(move_idx)
        moves_done += 1
        if not args.no_clear:
            clear()
        print(f"Score: {game.score}   Moves: {moves_done}   Last: {B.MOVE_NAMES[move_idx]}")
        print(game.render())
        if args.delay:
            time.sleep(args.delay)

    elapsed = time.perf_counter() - t_start
    if not args.no_clear:
        clear()
    print(f"GAME OVER - Score: {game.score}  Max tile: {game.max_tile()}  Moves: {moves_done}")
    print(game.render())
    print(f"(elapsed {elapsed:.1f}s, {elapsed / max(moves_done, 1):.3f}s/move avg)")


if __name__ == "__main__":
    main()
