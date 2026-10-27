"""Genetic algorithm that evolves the hybrid-search genome (see
ai2048/hybrid.py) by self-play fitness.

A genome is a length-10 float64 vector weighting the net value, the hand
heuristic, and MC rollouts inside expectimax (plus search-shape ints:
expectimax depth and rollout count/length). Fitness is measured by
playing a handful of games with evaluate_agent.evaluate and rewarding
average score plus (strongly) reaching the 16384 / 32768 tiles.

Usage:
    python ga.py --net runs/ms2_5t8 --pop 16 --elite 2 \
        --fitness-games 12 --generations 0 --out runs/ga [--resume]
"""
import argparse
import json
import os
import time

import numpy as np

from ai2048 import hybrid, netsearch, evaluate_agent

# --- fitness shaping constants ---------------------------------------
REACH_16384_BONUS = 50000.0
REACH_32768_BONUS = 200000.0
# Smooth signal: reward the AVERAGE max-tile exponent per game. Rare-tile
# reach rates (binary per game) are too coarse to optimize with few
# fitness games; avg exponent moves continuously (e.g. 13.2 -> 13.6 as
# the policy more often pushes 8192->16384), so the GA can sense progress
# with far less variance.
W_MAXEXP = 20000.0

# --- GA hyperparameters (per-gene) -----------------------------------
# Continuous genes: ALPHA/BETA/GAMMA mutate multiplicatively (log-space)
# since they can span orders of magnitude; the heuristic subweights and
# corner bonus mutate with gaussian noise scaled to their magnitude.
LOG_GENES = (hybrid.ALPHA, hybrid.BETA, hybrid.GAMMA)
# gaussian sigma (absolute) for the additive-mutation genes
ADD_SIGMA = {
    hybrid.W_EMPTY: 60.0,
    hybrid.W_MERGE: 150.0,
    hybrid.W_MONO: 15.0,
    hybrid.CORNER: 5000.0,
}
INT_GENES = (hybrid.DEPTH, hybrid.ROLL_K, hybrid.ROLL_L)
INT_RANGES = {hybrid.DEPTH: (0, 3), hybrid.ROLL_K: (0, 6), hybrid.ROLL_L: (1, 60)}

GENE_NAMES = [
    "ALPHA", "BETA", "GAMMA", "W_EMPTY", "W_MERGE", "W_MONO",
    "CORNER", "DEPTH", "ROLL_K", "ROLL_L",
]

MUT_PROB = 0.5          # per-gene chance to mutate a continuous gene
INT_NUDGE_PROB = 0.25   # per-int-gene chance to nudge by +/-1


def fitness_from_stats(stats, ms_per_move=0.0, time_penalty=0.0):
    """Scalar fitness from an evaluate() stats dict.

    Strength = avg score + big bonuses for reaching 16384/32768.
    Minus a cost term proportional to PER-MOVE compute time (ms/move) --
    NOT per-game time (stronger play makes longer games, so penalizing
    game time would punish strength). This pushes the GA toward efficient
    COMBINATIONS of methods rather than just brute-forcing search depth /
    rollout count. `time_penalty` = fitness points charged per ms/move.
    """
    reach = stats["reach"]
    f = (stats["avg_score"]
         + W_MAXEXP * stats.get("avg_max_exp", 0.0)
         + REACH_16384_BONUS * reach.get(16384, 0.0)
         + REACH_32768_BONUS * reach.get(32768, 0.0)
         - time_penalty * ms_per_move)
    return float(f)


def _avg_max_exp(stats):
    """Average max-tile exponent per game, from the tile histogram."""
    hist = stats.get("tile_hist", {})
    n = sum(hist.values())
    if n == 0:
        return 0.0
    tot = 0.0
    for tile, cnt in hist.items():
        tot += (int(tile).bit_length() - 1) * cnt   # log2(tile)
    return tot / n


def evaluate_genome(genome, w, thr, games, seed, time_penalty=0.0):
    import time as _time
    policy = hybrid.make_hybrid_policy(w, thr, genome, seed=seed + 777)
    t0 = _time.perf_counter()
    stats = evaluate_agent.evaluate(policy, games=games, seed=seed)
    wall = _time.perf_counter() - t0
    total_moves = games * max(stats.get("avg_moves", 0.0), 1.0)
    ms_per_move = wall * 1000.0 / total_moves
    stats["avg_max_exp"] = _avg_max_exp(stats)
    return fitness_from_stats(stats, ms_per_move=ms_per_move, time_penalty=time_penalty)


# --- population init --------------------------------------------------
def perturb(genome, rng):
    """Return a mutated, clamped copy of `genome`."""
    g = np.asarray(genome, dtype=np.float64).copy()
    for i in LOG_GENES:
        if rng.random() < MUT_PROB:
            # log-space multiplicative; allow turning a 0 gene on
            base = g[i] if g[i] > 1e-9 else 1e-3
            g[i] = base * float(np.exp(rng.normal(0.0, 0.7)))
    for i, sigma in ADD_SIGMA.items():
        if rng.random() < MUT_PROB:
            g[i] = g[i] + float(rng.normal(0.0, sigma))
    for i in INT_GENES:
        if rng.random() < INT_NUDGE_PROB:
            g[i] = g[i] + float(rng.choice([-1.0, 1.0]))
    return hybrid.clamp_genome(g)


def init_population(pop_size, rng):
    pop = [hybrid.clamp_genome(hybrid.DEFAULT_GENOME)]  # elite seed
    while len(pop) < pop_size:
        pop.append(perturb(hybrid.DEFAULT_GENOME, rng))
    return pop


# --- selection / crossover / mutation --------------------------------
def tournament(pop, fits, rng, k=3):
    idxs = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    best = idxs[0]
    for j in idxs[1:]:
        if fits[j] > fits[best]:
            best = j
    return pop[best]


def crossover(p1, p2, rng):
    """Blend crossover for continuous genes, uniform pick for int genes."""
    c = np.empty(hybrid.GENOME_LEN, dtype=np.float64)
    for i in range(hybrid.GENOME_LEN):
        if i in INT_RANGES:
            c[i] = p1[i] if rng.random() < 0.5 else p2[i]
        else:
            a = rng.random()  # BLX-ish blend
            c[i] = a * p1[i] + (1.0 - a) * p2[i]
    return c


def make_child(pop, fits, rng):
    p1 = tournament(pop, fits, rng)
    p2 = tournament(pop, fits, rng)
    child = crossover(p1, p2, rng)
    child = perturb(child, rng)
    return child


# --- checkpoint helpers ----------------------------------------------
def atomic_save_npy(path, arr):
    # Temp name already ends in .npy so np.save writes it verbatim (no
    # appended suffix to guess at) -- deterministic, and immune to a stale
    # "<path>.tmp" from a prior crash being promoted over the real output.
    # Mirrors train.py's _save_array. Callers pass paths ending in .npy.
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def genome_to_named(genome):
    return {name: float(genome[i]) for i, name in enumerate(GENE_NAMES)}


def save_best(out_dir, genome, fitness, gen):
    atomic_save_npy(os.path.join(out_dir, "best_genome.npy"),
                    np.asarray(genome, dtype=np.float64))
    obj = {"generation": gen, "fitness": float(fitness),
           "genome": genome_to_named(genome),
           "genome_vector": [float(x) for x in genome]}
    atomic_write_text(os.path.join(out_dir, "best_genome.json"),
                      json.dumps(obj, indent=2))


def save_pop(out_dir, pop, gen, seed):
    atomic_save_npy(os.path.join(out_dir, "pop.npy"),
                    np.asarray(pop, dtype=np.float64))
    atomic_write_text(os.path.join(out_dir, "ga_state.json"),
                      json.dumps({"generation": gen, "seed": seed}, indent=2))


def load_pop(out_dir):
    pop_path = os.path.join(out_dir, "pop.npy")
    state_path = os.path.join(out_dir, "ga_state.json")
    if not (os.path.exists(pop_path) and os.path.exists(state_path)):
        return None, None
    pop = [row.copy() for row in np.load(pop_path)]
    with open(state_path) as f:
        st = json.load(f)
    return pop, st.get("generation", 0)


def log_line(out_dir, text):
    print(text, flush=True)
    with open(os.path.join(out_dir, "ga.log"), "a") as f:
        f.write(text + "\n")


# --- main loop --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evolve the hybrid-search genome")
    ap.add_argument("--net", default="runs/ms2_5t8")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--generations", type=int, default=0, help="0 = forever")
    ap.add_argument("--fitness-games", type=int, default=12)
    ap.add_argument("--out", default="runs/ga")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-penalty", type=float, default=3000.0,
                    help="fitness points charged per ms/move of compute -- pushes the GA toward "
                         "efficient method COMBINATIONS instead of brute-forcing depth/rollouts. "
                         "0 = pure strength (will favor max depth).")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    w = netsearch.load_weights(args.net)
    thr = netsearch.load_thresholds(args.net)
    print(f"[ga] net={args.net} weights={w.shape} thresholds={thr.tolist()}", flush=True)

    rng = np.random.default_rng(args.seed)

    start_gen = 0
    pop = None
    if args.resume:
        pop, resumed_gen = load_pop(args.out)
        if pop is not None:
            start_gen = resumed_gen
            print(f"[ga] resumed {len(pop)} genomes at generation {start_gen}", flush=True)
    if pop is None:
        pop = init_population(args.pop, rng)

    gen = start_gen
    while args.generations == 0 or gen < start_gen + args.generations:
        t0 = time.time()
        # Common random numbers: the SAME seed-set across ALL generations
        # so fitness is directly comparable gen-to-gen and elitism truly
        # hill-climbs (the best never gets worse from seed luck). With a
        # noisy, few-games, rare-event fitness this matters far more than
        # the overfitting it risks -- we re-validate the winner on fresh
        # seeds at the end.
        gen_seed = (args.seed * 1000003 + 1) & 0x7FFFFFFF

        fits = np.array([
            evaluate_genome(g, w, thr, args.fitness_games, gen_seed, args.time_penalty)
            for g in pop
        ], dtype=np.float64)

        order = np.argsort(-fits)
        best_idx = int(order[0])
        best_genome = np.asarray(pop[best_idx], dtype=np.float64).copy()
        best_fit = float(fits[best_idx])

        save_best(args.out, best_genome, best_fit, gen)

        dt = time.time() - t0
        named = genome_to_named(best_genome)
        gstr = "  ".join(f"{k}={v:.4g}" for k, v in named.items())
        log_line(args.out,
                 f"gen {gen}  best_fit={best_fit:.1f}  "
                 f"mean_fit={float(np.mean(fits)):.1f}  "
                 f"time={dt:.1f}s  games={args.pop * args.fitness_games}")
        log_line(args.out, f"gen {gen}  best_genome: {gstr}")

        # --- build next generation ---
        n_elite = min(args.elite, len(pop))
        new_pop = [np.asarray(pop[int(order[i])], dtype=np.float64).copy()
                   for i in range(n_elite)]
        while len(new_pop) < args.pop:
            new_pop.append(make_child(pop, fits, rng))
        pop = new_pop

        gen += 1
        save_pop(args.out, pop, gen, args.seed)

    print("[ga] done", flush=True)


if __name__ == "__main__":
    main()
