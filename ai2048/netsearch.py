"""Inference with a trained N-tuple net: expectimax search using the
learned afterstate value as the leaf evaluation.

Two play modes:
  * 1-ply afterstate-greedy (`best_move_net` in trainkernel) -- fastest,
    the training-time policy.
  * depth-d expectimax whose leaves are scored by the N-tuple net
    (`best_move_search` here) -- strongest; the net makes a shallow
    search behave like a much deeper one.

The expectimax mirrors fastcore's max/chance structure but evaluates
leaves with `ntuple.evaluate` instead of the hand heuristic, and (like
fastcore) skips modelling 4-spawns when the board has many empties.
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict

from . import ntuple as NT
from .fastcore import move_left, move_right, move_up, move_down

# Transposition-table key/value types. Key = (board, depth*2 + node_type)
# packed as two uint64s; node_type 0 = chance node, 1 = max node. The TT
# caches the exact expectimax value of a (board, depth, node_type) subtree;
# the eval is deterministic given the weights so cached values are EXACT
# (no approximation) and reused whenever the same board recurs at the same
# remaining depth via a different move/spawn ordering -- which is common in
# the crowded late game, exactly where we want to search deeper.
_TT_KEY = types.UniTuple(types.uint64, 2)
_TT_VAL = types.float64


def _new_tt():
    """Fresh empty transposition table (call in Python, pass to the njit
    search)."""
    return Dict.empty(_TT_KEY, _TT_VAL)


@njit(cache=True)
def _count_empty(board):
    n = 0
    for i in range(16):
        if ((board >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:
            n += 1
    return n


@njit(cache=True)
def _max_exponent(board):
    m = 0
    for i in range(16):
        v = np.int64((board >> np.uint64(4 * i)) & np.uint64(0xF))
        if v > m:
            m = v
    return m


@njit(cache=True)
def _stage_of(board, thresholds):
    """Stage selector (mirrors trainkernel): count of max-tile-exponent
    thresholds the board meets/exceeds; empty thresholds -> stage 0."""
    n = thresholds.shape[0]
    if n == 0:
        return 0
    me = _max_exponent(board)
    s = 0
    for t in range(n):
        if me >= thresholds[t]:
            s += 1
    return s


# NOTE: _chance/_max are mutually recursive -> NO cache=True (numba
# segfaults on cached recursive functions when the on-disk cache is
# reloaded; same gotcha as fastcore's expectimax). best_move_search
# calls them so it stays uncached too.
@njit
def _chance(board, depth, weights, thresholds, sym):
    """Expected value over the random tile spawn, then recurse into the
    player's best move. `depth` counts remaining spawn layers. Leaves are
    valued at their own stage (multistage-aware)."""
    if depth <= 0:
        return NT.evaluate(board, weights, _stage_of(board, thresholds), sym)
    n = _count_empty(board)
    if n == 0:
        return NT.evaluate(board, weights, _stage_of(board, thresholds), sym)
    consider_four = n <= 6
    inv_n = 1.0 / n
    total = 0.0
    for i in range(16):
        if ((board >> np.uint64(4 * i)) & np.uint64(0xF)) != 0:
            continue
        b2 = board | (np.uint64(1) << np.uint64(4 * i))
        if consider_four:
            total += 0.9 * inv_n * _max(b2, depth, weights, thresholds, sym)
            b4 = board | (np.uint64(2) << np.uint64(4 * i))
            total += 0.1 * inv_n * _max(b4, depth, weights, thresholds, sym)
        else:
            total += inv_n * _max(b2, depth, weights, thresholds, sym)
    return total


@njit
def _max(board, depth, weights, thresholds, sym):
    """Best over the 4 moves of (reward + chance-value of afterstate)."""
    best = -1.0e30
    any_legal = False
    nb, sc, mv = move_left(board)
    if mv:
        any_legal = True
        v = sc + _chance(nb, depth - 1, weights, thresholds, sym)
        if v > best:
            best = v
    nb, sc, mv = move_right(board)
    if mv:
        any_legal = True
        v = sc + _chance(nb, depth - 1, weights, thresholds, sym)
        if v > best:
            best = v
    nb, sc, mv = move_up(board)
    if mv:
        any_legal = True
        v = sc + _chance(nb, depth - 1, weights, thresholds, sym)
        if v > best:
            best = v
    nb, sc, mv = move_down(board)
    if mv:
        any_legal = True
        v = sc + _chance(nb, depth - 1, weights, thresholds, sym)
        if v > best:
            best = v
    if not any_legal:
        return NT.evaluate(board, weights, _stage_of(board, thresholds), sym) - 200000.0
    return best


@njit
def best_move_search(board, depth, weights, thresholds, sym):
    """Move index (0=L,1=R,2=U,3=D) maximizing reward + expectimax value
    of the afterstate, searching `depth` spawn layers deep. -1 if no
    legal move. depth=0 reduces to 1-ply afterstate-greedy."""
    best = -1.0e30
    best_idx = -1
    nb, sc, mv = move_left(board)
    if mv:
        v = sc + _chance(nb, depth, weights, thresholds, sym)
        if v > best:
            best = v; best_idx = 0
    nb, sc, mv = move_right(board)
    if mv:
        v = sc + _chance(nb, depth, weights, thresholds, sym)
        if v > best:
            best = v; best_idx = 1
    nb, sc, mv = move_up(board)
    if mv:
        v = sc + _chance(nb, depth, weights, thresholds, sym)
        if v > best:
            best = v; best_idx = 2
    nb, sc, mv = move_down(board)
    if mv:
        v = sc + _chance(nb, depth, weights, thresholds, sym)
        if v > best:
            best = v; best_idx = 3
    return best_idx


# --- transposition-table variants ------------------------------------
# Same expectimax as _chance/_max above, but memoize each subtree value in
# a shared typed.Dict `tt`. NO cache=True (recursive + typed.Dict arg ->
# numba segfault footgun; same reason the plain versions stay uncached).
@njit
def _chance_tt(board, depth, weights, thresholds, sym, tt):
    if depth <= 0:
        return NT.evaluate(board, weights, _stage_of(board, thresholds), sym)
    n = _count_empty(board)
    if n == 0:
        return NT.evaluate(board, weights, _stage_of(board, thresholds), sym)
    key = (board, np.uint64(depth) * np.uint64(2) + np.uint64(0))
    if key in tt:
        return tt[key]
    consider_four = n <= 6
    inv_n = 1.0 / n
    total = 0.0
    for i in range(16):
        if ((board >> np.uint64(4 * i)) & np.uint64(0xF)) != 0:
            continue
        b2 = board | (np.uint64(1) << np.uint64(4 * i))
        if consider_four:
            total += 0.9 * inv_n * _max_tt(b2, depth, weights, thresholds, sym, tt)
            b4 = board | (np.uint64(2) << np.uint64(4 * i))
            total += 0.1 * inv_n * _max_tt(b4, depth, weights, thresholds, sym, tt)
        else:
            total += inv_n * _max_tt(b2, depth, weights, thresholds, sym, tt)
    tt[key] = total
    return total


@njit
def _max_tt(board, depth, weights, thresholds, sym, tt):
    key = (board, np.uint64(depth) * np.uint64(2) + np.uint64(1))
    if key in tt:
        return tt[key]
    best = -1.0e30
    any_legal = False
    nb, sc, mv = move_left(board)
    if mv:
        any_legal = True
        v = sc + _chance_tt(nb, depth - 1, weights, thresholds, sym, tt)
        if v > best:
            best = v
    nb, sc, mv = move_right(board)
    if mv:
        any_legal = True
        v = sc + _chance_tt(nb, depth - 1, weights, thresholds, sym, tt)
        if v > best:
            best = v
    nb, sc, mv = move_up(board)
    if mv:
        any_legal = True
        v = sc + _chance_tt(nb, depth - 1, weights, thresholds, sym, tt)
        if v > best:
            best = v
    nb, sc, mv = move_down(board)
    if mv:
        any_legal = True
        v = sc + _chance_tt(nb, depth - 1, weights, thresholds, sym, tt)
        if v > best:
            best = v
    if not any_legal:
        best = NT.evaluate(board, weights, _stage_of(board, thresholds), sym) - 200000.0
    tt[key] = best
    return best


@njit
def best_move_search_tt(board, depth, weights, thresholds, sym, tt):
    """TT-backed version of best_move_search. `tt` is a fresh typed.Dict
    (one per decision). Returns an IDENTICAL move to best_move_search for
    the same (board, depth) -- the TT only avoids recomputing repeated
    subtrees, it does not change the values."""
    best = -1.0e30
    best_idx = -1
    nb, sc, mv = move_left(board)
    if mv:
        v = sc + _chance_tt(nb, depth, weights, thresholds, sym, tt)
        if v > best:
            best = v; best_idx = 0
    nb, sc, mv = move_right(board)
    if mv:
        v = sc + _chance_tt(nb, depth, weights, thresholds, sym, tt)
        if v > best:
            best = v; best_idx = 1
    nb, sc, mv = move_up(board)
    if mv:
        v = sc + _chance_tt(nb, depth, weights, thresholds, sym, tt)
        if v > best:
            best = v; best_idx = 2
    nb, sc, mv = move_down(board)
    if mv:
        v = sc + _chance_tt(nb, depth, weights, thresholds, sym, tt)
        if v > best:
            best = v; best_idx = 3
    return best_idx


def load_weights(ckpt_dir):
    """Load just the weights array from a training checkpoint dir."""
    import os
    return np.load(os.path.join(ckpt_dir, "weights.npy"))


def load_thresholds(ckpt_dir):
    """Read the stage thresholds recorded in a checkpoint's meta.json
    (empty array if single-stage or absent)."""
    import os, json
    p = os.path.join(ckpt_dir, "meta.json")
    thr = []
    if os.path.exists(p):
        with open(p) as f:
            thr = json.load(f).get("stage_thresholds", []) or []
    return np.array(thr, dtype=np.int64)


def best_move_timed(board, weights, thresholds, sym, time_limit=0.15, min_depth=0,
                    max_depth=6, use_tt=False):
    """Time-budgeted iterative-deepening net search. Goes as deep as the
    per-move budget allows -- which is shallow in the open early game
    (large chance-node branching) and automatically DEEP in the crowded
    late game (few empty cells => tiny branching, so extra plies are
    cheap AND most valuable). Directly targets late-game strength.

    With `use_tt=True`, each depth pass memoizes repeated subtrees in a
    fresh transposition table, so the same time budget reaches deeper
    (more plies) in the crowded late game -- a pure-algorithm speedup.

    Returns move idx 0..3, or None if no legal move.
    """
    import time
    board_u = np.uint64(board)

    def _search(d):
        if use_tt:
            return best_move_search_tt(board_u, d, weights, thresholds, sym, _new_tt())
        return best_move_search(board_u, d, weights, thresholds, sym)

    # always get at least the min_depth result
    best = _search(min_depth)
    if best < 0:
        return None
    t_start = time.perf_counter()
    elapsed = []
    depth = min_depth
    SAFETY = 1.6
    while depth < max_depth:
        remaining = time_limit - (time.perf_counter() - t_start)
        if remaining < 0.002:
            break
        if len(elapsed) >= 2 and elapsed[-2] > 0.0:
            growth = max(elapsed[-1] / elapsed[-2], 2.0)
        else:
            growth = 12.0
        if elapsed and elapsed[-1] * growth * SAFETY > remaining:
            break
        depth += 1
        t0 = time.perf_counter()
        mv = _search(depth)
        elapsed.append(time.perf_counter() - t0)
        if mv >= 0:
            best = mv
        if time.perf_counter() - t_start > time_limit:
            break
    return int(best)


def make_policy(weights, depth=1, thresholds=None, time_limit=None, min_depth=0,
                max_depth=6, use_tt=False):
    """Return a `policy(board_int) -> move_idx|None` for the eval harness.

    If `time_limit` is set (seconds), use time-budgeted iterative deepening
    (`best_move_timed`) -- shallow early, deep in the crowded late game.
    Otherwise fixed `depth`: 0 -> 1-ply greedy; depth>=1 -> expectimax with
    net leaves (number of spawn/chance layers). `thresholds` is the
    multistage boundary array (None/empty -> single stage). `use_tt` enables
    the transposition table (identical play, deeper reach per time budget).
    """
    sym = NT.SYM_PATTERNS
    thr = np.array([], dtype=np.int64) if thresholds is None else np.asarray(thresholds, dtype=np.int64)

    if time_limit is not None:
        def policy(board):
            return best_move_timed(board, weights, thr, sym, time_limit=time_limit,
                                   min_depth=min_depth, max_depth=max_depth, use_tt=use_tt)
        return policy

    if use_tt:
        def policy(board):
            mv = best_move_search_tt(np.uint64(board), depth, weights, thr, sym, _new_tt())
            return None if mv < 0 else int(mv)
        return policy

    def policy(board):
        mv = best_move_search(np.uint64(board), depth, weights, thr, sym)
        return None if mv < 0 else int(mv)

    return policy
