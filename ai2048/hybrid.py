"""Hybrid 2048 search: combines a trained N-tuple net, the hand
heuristic, and truncated Monte-Carlo rollouts inside an expectimax, all
driven by a single float64 *genome* vector so a genetic algorithm (ga.py)
can evolve how the three signals are combined.

Board representation is the project standard (see fastcore.py / ntuple.py):
a uint64 of 16 cells x 4 bits, cell = tile exponent, row-major.

== Leaf value ==
Every expectimax leaf is scored by `leaf_value` =
    ALPHA * net_value(board) + BETA * hand_heuristic(board)
(net = ntuple.evaluate at the board's own stage; heuristic =
fastcore.evaluate with genome-supplied subweights). This blended leaf is
used EVERYWHERE a leaf is needed -- never the net alone.

== Expectimax ==
`_chance` / `_max` mirror netsearch.py's structure exactly, including its
depth convention (best_move_search / _chance pass the SAME depth into
_max, and _max decrements it). They are MUTUALLY RECURSIVE, so per the
fastcore.py warning they are plain @njit with NO cache=True (cached
recursive njit funcs segfault when the on-disk cache reloads).

== Rollouts ==
`_rollout` is a truncated Monte-Carlo rollout from a post-move afterstate
under a fast net-1-ply-greedy policy. Rollouts are run at the ROOT only
(<= 4*K total per decision), NEVER from inside _chance/_max leaves, so the
cost stays bounded.
"""
import numpy as np
from numba import njit

from . import ntuple as NT
from .fastcore import move_left, move_right, move_up, move_down, evaluate as _heur_eval
from .netsearch import _stage_of, _count_empty
from .trainkernel import _spawn

# --- Genome layout ----------------------------------------------------
GENOME_LEN = 10
ALPHA = 0    # weight on net value            (>=0)
BETA = 1     # weight on hand heuristic        (>=0)
GAMMA = 2    # weight on rollout value         (>=0)
W_EMPTY = 3  # heuristic subweight
W_MERGE = 4  # heuristic subweight
W_MONO = 5   # heuristic subweight
CORNER = 6   # heuristic corner bonus
DEPTH = 7    # expectimax chance layers (round->int, clamp 0..3)
ROLL_K = 8   # rollouts per root move  (round->int, clamp 0..6; 0 disables)
ROLL_L = 9   # rollout length cap in moves (round->int, clamp 1..60)

# Reasonable starting point: ~pure net, GA can switch on heuristic/rollout.
DEFAULT_GENOME = np.array(
    [1.0, 0.0001, 0.0, 270.0, 700.0, 47.0, 20000.0, 1.0, 0.0, 20.0],
    dtype=np.float64,
)

GAME_OVER_PENALTY = 200000.0


# --- genome cleaning (Python) -----------------------------------------
def clamp_genome(g):
    """Return a cleaned float64 copy of genome `g`: non-negative
    ALPHA/BETA/GAMMA; DEPTH/ROLL_K/ROLL_L rounded to ints and clamped to
    their valid ranges. Used by both the policy and the GA."""
    g = np.asarray(g, dtype=np.float64).copy()
    if g.shape[0] != GENOME_LEN:
        raise ValueError(f"genome must have length {GENOME_LEN}, got {g.shape[0]}")
    if g[ALPHA] < 0.0:
        g[ALPHA] = 0.0
    if g[BETA] < 0.0:
        g[BETA] = 0.0
    if g[GAMMA] < 0.0:
        g[GAMMA] = 0.0
    g[DEPTH] = float(min(3, max(0, int(round(g[DEPTH])))))
    g[ROLL_K] = float(min(6, max(0, int(round(g[ROLL_K])))))
    g[ROLL_L] = float(min(60, max(1, int(round(g[ROLL_L])))))
    return g


# --- blended leaf value (non-recursive -> cache OK) -------------------
@njit(cache=True)
def leaf_value(board, weights, thresholds, sym, g):
    net = NT.evaluate(board, weights, _stage_of(board, thresholds), sym)
    heur = _heur_eval(board, g[W_EMPTY], g[W_MERGE], g[W_MONO], g[CORNER], 0.0)
    return g[ALPHA] * net + g[BETA] * heur


# --- expectimax (mutually recursive -> NO cache=True) -----------------
@njit
def _chance(board, depth, weights, thresholds, sym, g):
    """Expected value over the random tile spawn, recursing into the
    player's best move. `depth` counts remaining spawn layers. Mirrors
    netsearch._chance; leaf is the blended leaf_value."""
    if depth <= 0:
        return leaf_value(board, weights, thresholds, sym, g)
    n = _count_empty(board)
    if n == 0:
        return leaf_value(board, weights, thresholds, sym, g)
    consider_four = n <= 6
    inv_n = 1.0 / n
    total = 0.0
    for i in range(16):
        if ((board >> np.uint64(4 * i)) & np.uint64(0xF)) != 0:
            continue
        b2 = board | (np.uint64(1) << np.uint64(4 * i))
        if consider_four:
            total += 0.9 * inv_n * _max(b2, depth, weights, thresholds, sym, g)
            b4 = board | (np.uint64(2) << np.uint64(4 * i))
            total += 0.1 * inv_n * _max(b4, depth, weights, thresholds, sym, g)
        else:
            total += inv_n * _max(b2, depth, weights, thresholds, sym, g)
    return total


@njit
def _max(board, depth, weights, thresholds, sym, g):
    """Best over the 4 moves of (reward + chance-value of afterstate).
    Mirrors netsearch._max; decrements depth into _chance."""
    best = -1.0e30
    any_legal = False
    nb, sc, mv = move_left(board)
    if mv:
        any_legal = True
        v = sc + _chance(nb, depth - 1, weights, thresholds, sym, g)
        if v > best:
            best = v
    nb, sc, mv = move_right(board)
    if mv:
        any_legal = True
        v = sc + _chance(nb, depth - 1, weights, thresholds, sym, g)
        if v > best:
            best = v
    nb, sc, mv = move_up(board)
    if mv:
        any_legal = True
        v = sc + _chance(nb, depth - 1, weights, thresholds, sym, g)
        if v > best:
            best = v
    nb, sc, mv = move_down(board)
    if mv:
        any_legal = True
        v = sc + _chance(nb, depth - 1, weights, thresholds, sym, g)
        if v > best:
            best = v
    if not any_legal:
        return leaf_value(board, weights, thresholds, sym, g) - GAME_OVER_PENALTY
    return best


# --- truncated Monte-Carlo rollout (non-recursive -> cache OK) --------
@njit(cache=True)
def _rollout(afterstate, Lcap, state, weights, thresholds, sym, g):
    """Truncated MC rollout from a post-move `afterstate`. Spawn a random
    tile, then play a net-1-ply-greedy policy for up to Lcap moves (or
    until no legal move), summing merge rewards. Add a bootstrap
    leaf_value(final_board) so truncation isn't biased. Threads and
    returns the xorshift `state`. Returns (total_float, new_state)."""
    total = 0.0
    board, state = _spawn(afterstate, state)
    moves = 0
    while moves < Lcap:
        best_val = -1.0e30
        best_after = np.uint64(0)
        best_reward = 0
        any_legal = False

        nb, sc, mv = move_left(board)
        if mv:
            any_legal = True
            v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym)
            if v > best_val:
                best_val = v; best_after = nb; best_reward = sc
        nb, sc, mv = move_right(board)
        if mv:
            any_legal = True
            v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym)
            if v > best_val:
                best_val = v; best_after = nb; best_reward = sc
        nb, sc, mv = move_up(board)
        if mv:
            any_legal = True
            v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym)
            if v > best_val:
                best_val = v; best_after = nb; best_reward = sc
        nb, sc, mv = move_down(board)
        if mv:
            any_legal = True
            v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym)
            if v > best_val:
                best_val = v; best_after = nb; best_reward = sc

        if not any_legal:
            break
        total += best_reward
        board, state = _spawn(best_after, state)
        moves += 1

    total += leaf_value(board, weights, thresholds, sym, g)
    return total, state


# --- root move selection -----------------------------------------------
# NOTE: although hybrid_best_move is not itself recursive, it CALLS the
# mutually-recursive _chance/_max. A cache=True function that reaches the
# recursive chain triggers numba's cached-recursion segfault on cache
# reload (same gotcha as fastcore). So this stays plain @njit (no cache).
@njit
def hybrid_best_move(board, weights, thresholds, sym, g, state):
    """Pick the best move (0=L,1=R,2=U,3=D; -1 if none) combining the
    expectimax value (net+heuristic leaves) with root-only MC rollouts.
    Returns (best_idx, new_state). Rollouts are run only here, never in
    the expectimax leaves."""
    depth = int(round(g[DEPTH]))
    if depth < 0:
        depth = 0
    elif depth > 3:
        depth = 3
    K = int(round(g[ROLL_K]))
    if K < 0:
        K = 0
    elif K > 6:
        K = 6
    Lcap = int(round(g[ROLL_L]))
    if Lcap < 1:
        Lcap = 1
    elif Lcap > 60:
        Lcap = 60

    use_roll = K > 0 and g[GAMMA] != 0.0

    best = -1.0e30
    best_idx = -1

    for idx in range(4):
        if idx == 0:
            nb, sc, mv = move_left(board)
        elif idx == 1:
            nb, sc, mv = move_right(board)
        elif idx == 2:
            nb, sc, mv = move_up(board)
        else:
            nb, sc, mv = move_down(board)
        if not mv:
            continue
        v = sc + _chance(nb, depth, weights, thresholds, sym, g)
        if use_roll:
            racc = 0.0
            for _k in range(K):
                rv, state = _rollout(nb, Lcap, state, weights, thresholds, sym, g)
                racc += rv
            v += g[GAMMA] * (racc / K)
        if v > best:
            best = v
            best_idx = idx

    return best_idx, state


# --- Python policy wrapper --------------------------------------------
def make_hybrid_policy(weights, thresholds, g, seed=12345):
    """Return `policy(board_int) -> move_idx(0..3) | None` for the eval
    harness. Holds a mutable uint64 xorshift state threaded through
    hybrid_best_move. Genome `g` is cleaned once via clamp_genome."""
    gc = clamp_genome(g)
    sym = NT.SYM_PATTERNS
    thr = np.asarray(thresholds, dtype=np.int64)
    _MASK = 0xFFFFFFFFFFFFFFFF
    s0 = int(seed) & _MASK
    if s0 == 0:
        s0 = 0x9E3779B97F4A7C15
    state = [np.uint64(s0)]

    def policy(board):
        # Mask to 64 bits before np.uint64() -- numpy 2.x raises
        # OverflowError on any Python int outside [0, 2**64).
        b = np.uint64(int(board) & _MASK)
        st = np.uint64(int(state[0]) & _MASK)
        idx, new_state = hybrid_best_move(b, weights, thr, sym, gc, st)
        state[0] = np.uint64(int(new_state) & _MASK)
        return None if idx < 0 else int(idx)

    return policy


# --- warmup: JIT-compile everything on a tiny board at import ----------
def _warmup():
    try:
        tiny_w = np.zeros((1, NT.NUM_BASE, 16), dtype=np.float32)
        sym = NT.SYM_PATTERNS
        thr = np.array([], dtype=np.int64)
        b0 = np.uint64(0)
        # a tiny genome exercising depth + rollouts so every kernel compiles
        gw = clamp_genome(DEFAULT_GENOME)
        gw[GAMMA] = 0.3
        gw[DEPTH] = 1.0
        gw[ROLL_K] = 1.0
        gw[ROLL_L] = 2.0
        st = np.uint64(0x9E3779B97F4A7C15)
        leaf_value(b0, tiny_w, thr, sym, gw)
        _rollout(b0, 2, st, tiny_w, thr, sym, gw)
        hybrid_best_move(b0, tiny_w, thr, sym, gw, st)
    except Exception:
        pass


_warmup()
