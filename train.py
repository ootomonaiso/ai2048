"""Train the 2048 N-tuple network by TD-afterstate(+TC) self-play.

Designed for weeks-long background runs: checkpoints (weights + TC
accumulators + metadata) are saved atomically at a fixed interval and
training resumes from the latest checkpoint automatically. Progress is
appended to a log with timestamps and periodic greedy-play evaluation
(max-tile reach rates) so the learning curve is visible at any time.

Usage:
    python train.py                       # start/resume with defaults
    python train.py --ckpt runs/exp1      # checkpoint into a dir
    python train.py --lr 1.0 --batch 256 --eval-every 50000 --eval-games 100
    python train.py --total 0             # run forever (until killed)

Interrupt (Ctrl-C) any time; the latest checkpoint is safe to resume.
"""
import argparse
import json
import os
import time
import datetime

import numpy as np

from ai2048 import ntuple as NT
from ai2048 import trainkernel as TK
from ai2048.evaluate_agent import evaluate as eval_agent, format_stats


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _save_array(ckpt_dir, name, arr):
    path = os.path.join(ckpt_dir, name + ".npy")
    tmp = os.path.join(ckpt_dir, name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_ckpt(ckpt_dir, weights, E, A, meta):
    os.makedirs(ckpt_dir, exist_ok=True)
    _save_array(ckpt_dir, "weights", weights)
    _save_array(ckpt_dir, "E", E)
    _save_array(ckpt_dir, "A", A)
    meta_path = os.path.join(ckpt_dir, "meta.json")
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, meta_path)


def load_ckpt(ckpt_dir):
    meta_path = os.path.join(ckpt_dir, "meta.json")
    wpath = os.path.join(ckpt_dir, "weights.npy")
    if not (os.path.exists(meta_path) and os.path.exists(wpath)):
        return None
    weights = np.load(wpath)
    E = np.load(os.path.join(ckpt_dir, "E.npy"))
    A = np.load(os.path.join(ckpt_dir, "A.npy"))
    with open(meta_path) as f:
        meta = json.load(f)
    return weights, E, A, meta


def log_line(ckpt_dir, msg):
    print(msg, flush=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "train.log"), "a") as f:
        f.write(msg + "\n")


def net_policy(weights, thresholds):
    sym = NT.SYM_PATTERNS
    def policy(board):
        mv = TK.best_move_net(np.uint64(board), weights, thresholds, sym)
        return None if mv < 0 else int(mv)
    return policy


def main():
    parser = argparse.ArgumentParser(description="Train the 2048 N-tuple net (TD-afterstate+TC)")
    parser.add_argument("--ckpt", default="runs/ntuple1", help="checkpoint directory")
    parser.add_argument("--lr", type=float, default=0.1, help="base learning rate (TC on top); "
                        "0.1 validated to learn well, 1.0 diverges (chases spawn noise)")
    parser.add_argument("--batch", type=int, default=256, help="self-play games per parallel batch")
    parser.add_argument("--ckpt-every", type=int, default=100000, help="games between checkpoints")
    parser.add_argument("--eval-every", type=int, default=50000, help="games between evaluations")
    parser.add_argument("--eval-games", type=int, default=100, help="games per evaluation")
    parser.add_argument("--total", type=int, default=0, help="stop after this many games (0 = forever)")
    parser.add_argument("--stage-thresholds", default="", help="comma-separated max-tile EXPONENT "
                        "thresholds for multistage, e.g. '13,14,15' -> 4 stages (0 below 8192, 1 at "
                        "8192, 2 at 16384, 3 at >=32768). Empty = single stage.")
    parser.add_argument("--init-from", default="", help="warm-start a fresh run by broadcasting this "
                        "checkpoint dir's stage-0 weights/TC into every stage (same tuple profile "
                        "required). Ignored when resuming an existing --ckpt.")
    args = parser.parse_args()

    thresholds = np.array(
        [int(x) for x in args.stage_thresholds.split(",") if x.strip()],
        dtype=np.int64,
    )
    num_stages = len(thresholds) + 1

    ckpt = load_ckpt(args.ckpt)
    if ckpt is not None:
        weights, E, A, meta = ckpt
        # resume must use the SAME stage layout the checkpoint was built with
        saved = np.array(meta.get("stage_thresholds", []), dtype=np.int64)
        if saved.shape[0] + 1 != weights.shape[0]:
            saved = np.array([], dtype=np.int64)  # legacy single-stage ckpt
        thresholds = saved
        num_stages = weights.shape[0]
        log_line(args.ckpt, f"[{_now()}] resumed from {args.ckpt} at {meta['games']} games "
                            f"({num_stages} stage(s), thresholds={thresholds.tolist()})")
    else:
        weights = NT.make_weights(num_stages)
        E, A = NT.make_tc_accumulators(num_stages)
        meta = {"games": 0, "wall_seconds": 0.0, "lr": args.lr,
                "stage_thresholds": thresholds.tolist()}
        if args.init_from:
            # Warm start: broadcast a source checkpoint's stage-0 weights
            # (and TC accumulators) into EVERY stage of this new net, so a
            # multistage run inherits a strong single-stage net rather than
            # relearning the early game from scratch. Requires matching
            # tuple profile (same per-stage shape).
            sw = np.load(os.path.join(args.init_from, "weights.npy"))
            se = np.load(os.path.join(args.init_from, "E.npy"))
            sa = np.load(os.path.join(args.init_from, "A.npy"))
            if sw.shape[1:] != weights.shape[1:]:
                raise ValueError(f"--init-from shape {sw.shape} incompatible with "
                                 f"target {weights.shape} (different tuple profile?)")
            for st in range(num_stages):
                weights[st] = sw[0]
                E[st] = se[0]
                A[st] = sa[0]
            log_line(args.ckpt, f"[{_now()}] warm-started all {num_stages} stage(s) "
                                f"from {args.init_from} (stage 0 broadcast)")
        log_line(args.ckpt, f"[{_now()}] fresh start: {weights.shape} weights "
                            f"({weights.nbytes / 1e9:.2f} GB), lr={args.lr}, "
                            f"{num_stages} stage(s), thresholds={thresholds.tolist()}")

    sym = NT.SYM_PATTERNS
    out_score = np.zeros(args.batch, dtype=np.int64)
    out_maxexp = np.zeros(args.batch, dtype=np.int64)

    games = meta["games"]
    last_ckpt = games
    last_eval = games
    seed_counter = np.uint64(games * 2654435761 + 1)  # continue the seed stream
    t_wall0 = time.perf_counter()
    base_wall = meta.get("wall_seconds", 0.0)

    recent_scores = []
    recent_maxexp = []

    log_line(args.ckpt, f"[{_now()}] warming up JIT (first batch compiles)...")

    try:
        while args.total == 0 or games < args.total:
            # build nonzero seeds (uint64 LCG; wraparound overflow is intended)
            seeds = np.empty(args.batch, dtype=np.uint64)
            with np.errstate(over="ignore"):
                for i in range(args.batch):
                    seed_counter = (seed_counter * np.uint64(6364136223846793005)
                                    + np.uint64(1442695040888963407))
                    s = seed_counter ^ (seed_counter >> np.uint64(29))
                    seeds[i] = s if s != np.uint64(0) else np.uint64(1)

            TK.train_batch(seeds, weights, E, A, thresholds, sym, args.lr, out_score, out_maxexp)
            games += args.batch
            recent_scores.extend(out_score.tolist())
            recent_maxexp.extend(out_maxexp.tolist())
            if len(recent_scores) > 5000:
                recent_scores = recent_scores[-5000:]
                recent_maxexp = recent_maxexp[-5000:]

            # periodic evaluation
            if games - last_eval >= args.eval_every:
                last_eval = games
                tr_avg = sum(recent_scores) / len(recent_scores)
                tr_maxtile = 1 << max(recent_maxexp)
                stats = eval_agent(net_policy(weights, thresholds), games=args.eval_games,
                                   seed=1_000_000)
                gps = args.batch / max(1e-9, (time.perf_counter() - t_wall0)) if False else None
                log_line(args.ckpt,
                         f"[{_now()}] games={games:,} train_avg={tr_avg:.0f} "
                         f"train_best_tile={tr_maxtile} | EVAL "
                         f"avg={stats['avg_score']:.0f} best={stats['best_score']} "
                         f"2048={stats['reach'][2048]*100:.0f}% "
                         f"4096={stats['reach'][4096]*100:.0f}% "
                         f"8192={stats['reach'][8192]*100:.0f}% "
                         f"16384={stats['reach'][16384]*100:.0f}% "
                         f"32768={stats['reach'][32768]*100:.0f}%")

            # periodic checkpoint
            if games - last_ckpt >= args.ckpt_every:
                last_ckpt = games
                meta["games"] = games
                meta["wall_seconds"] = base_wall + (time.perf_counter() - t_wall0)
                meta["lr"] = args.lr
                meta["stage_thresholds"] = thresholds.tolist()
                save_ckpt(args.ckpt, weights, E, A, meta)
                log_line(args.ckpt, f"[{_now()}] checkpoint saved at {games:,} games "
                                    f"({meta['wall_seconds']/3600:.2f}h wall)")
    except KeyboardInterrupt:
        log_line(args.ckpt, f"[{_now()}] interrupted; saving final checkpoint...")

    meta["games"] = games
    meta["wall_seconds"] = base_wall + (time.perf_counter() - t_wall0)
    meta["lr"] = args.lr
    meta["stage_thresholds"] = thresholds.tolist()
    save_ckpt(args.ckpt, weights, E, A, meta)
    log_line(args.ckpt, f"[{_now()}] final checkpoint saved at {games:,} games")


if __name__ == "__main__":
    main()
