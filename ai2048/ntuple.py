"""N-tuple network value function for the 2048 AI.

This replaces the hand-written heuristic in fastcore.py with a learned
value function (trained elsewhere by TD-afterstate + TC self-play). This
module owns ONLY the network *representation*: the tuple patterns, the
8-fold board symmetry, feature indexing, weight/accumulator storage, and
the njit evaluate / TC-update kernels. The training loop lives in a
separate module and calls the njit functions here.

Board representation (same as tables.py / fastcore.py): a 64-bit int,
16 cells of 4 bits each, cell value = tile exponent (0 = empty, n = tile
2**n). Row-major: cell index = row*4 + col, bits [4*idx, 4*idx+4).

==================  KEY CONVENTIONS (read before using)  ==================

* TILE CAP: cell exponents are clamped to 15 when forming feature
  indices (TILE_CAP = 15). Because cells are physically 4 bits they can
  never exceed 15 anyway, so this engine caps at the 32768 tile
  (exponent 15). That matches the project's SOTA-32768 target. The
  min(v, 15) in feature_index is therefore defensive and effectively a
  no-op on any real board.

* SYMMETRY: the 4x4 board has 8 symmetries (dihedral group D4 = 4
  rotations x 2 reflections). SYM_PERM has shape (8, 16);
  SYM_PERM[s][old_idx] = new_idx, i.e. the cell that sits at old_idx
  moves to new_idx under symmetry s. SYM_PERM[0] is the identity.
  A pattern's 8 symmetric variants are obtained by mapping each of its
  cell indices through SYM_PERM[s]; this yields the 8 oriented copies of
  the tuple. We KEEP ALL 8 variants per base pattern even if a pattern
  is self-symmetric and some variants coincide (none of the default
  patterns are self-symmetric, but the rule is uniform). Consequently
  every evaluate()/update() touches exactly NUM_BASE*8 weight entries,
  and the learner must spread one TD error evenly across all of them
  (delta_share = delta / (NUM_BASE*8)); see update().

* WEIGHT MEMORY: weights have shape (NUM_STAGES, NUM_BASE, 16**6), dtype
  float32. 16**6 = 16,777,216 entries per (stage, tuple). One stage =
  NUM_BASE(8) * 16.7M * 4 bytes ~= 0.5 GB. NUM_STAGES stages multiply
  that. Allocate deliberately. The TC accumulators E and A each have the
  same shape, so a single-stage TC setup is ~1.5 GB total (weights + E +
  A). Plan RAM accordingly before going multistage.

===========================================================================
"""
import os

import numpy as np
from numba import njit

# --- Tuple patterns ---------------------------------------------------
# Cell indices are 0..15 row-major:
#     0  1  2  3
#     4  5  6  7
#     8  9 10 11
#    12 13 14 15
# Two profiles, selected by env var AI2048_NTUPLE (default "5t8"):
#   "5t8" -- eight 5-tuples. INDEX_SIZE=16^5=1.05M -> ~4MB/tuple/array,
#            ~96 MB total (weights+E+A) single stage. Fits small RAM and
#            multistage comfortably. Strong, slightly below the 6-tuple
#            ceiling. DEFAULT.
#   "6t8" -- the standard strong eight 6-tuples. INDEX_SIZE=16^6=16.8M ->
#            ~64MB/tuple/array, ~1.6 GB total single stage. Higher ceiling
#            (SOTA configs use 6-tuples) but needs real RAM, especially
#            multistage. Switch to this for the big run once RAM is freed.
# Shapes are swappable tuning knobs, not sacred. Changing the profile (or
# the shapes) changes the weight-array shape and INVALIDATES existing
# checkpoints -- start a fresh run dir when you switch.
_PROFILES = {
    "5t8": [
        [0, 1, 2, 3, 4],
        [4, 5, 6, 7, 8],
        [0, 1, 2, 4, 5],
        [1, 2, 3, 5, 6],
        [4, 5, 6, 8, 9],
        [5, 6, 7, 9, 10],
        [0, 1, 4, 5, 8],
        [1, 2, 5, 6, 9],
    ],
    "6t8": [
        [0, 1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8, 9],
        [0, 1, 2, 4, 5, 6],
        [4, 5, 6, 8, 9, 10],
        [0, 1, 4, 5, 8, 9],
        [1, 2, 5, 6, 9, 10],
        [0, 1, 2, 5, 6, 10],
        [0, 1, 5, 6, 7, 10],
    ],
}
NTUPLE_PROFILE = os.environ.get("AI2048_NTUPLE", "5t8")
if NTUPLE_PROFILE not in _PROFILES:
    raise ValueError(f"Unknown AI2048_NTUPLE profile {NTUPLE_PROFILE!r}; "
                     f"choose from {sorted(_PROFILES)}")
BASE_PATTERNS = _PROFILES[NTUPLE_PROFILE]
NUM_BASE = len(BASE_PATTERNS)
TUPLE_LEN = len(BASE_PATTERNS[0])
assert all(len(p) == TUPLE_LEN for p in BASE_PATTERNS), "all tuples must share one length"
NUM_SYM = 8
TILE_CAP = 15

# --- Multistage config ------------------------------------------------
# A "stage" is a slice of the game selected (by the learner) from the
# current board, typically via max-tile thresholds. Storage supports an
# arbitrary number of stages; start at 1. The stage-selection function
# itself belongs to the training module.
NUM_STAGES = 1

# Feature-index space per (stage, tuple).
INDEX_SIZE = 16 ** TUPLE_LEN  # 16,777,216


def _build_sym_perm():
    """8 D4 symmetries as cell-index permutations of shape (8, 16).

    SYM_PERM[s][old_idx] = new_idx: the cell at old_idx moves to new_idx
    under symmetry s. Generated programmatically by composing rotations
    with an optional horizontal flip.
    """
    def rot90(r, c):            # 90 deg clockwise
        return c, 3 - r

    def flip_h(r, c):           # mirror columns
        return r, 3 - c

    perms = []
    for k in range(4):          # number of 90-deg CW rotations
        for f in range(2):      # reflect first or not
            perm = np.empty(16, dtype=np.int64)
            for old in range(16):
                r, c = divmod(old, 4)
                if f:
                    r, c = flip_h(r, c)
                for _ in range(k):
                    r, c = rot90(r, c)
                perm[old] = r * 4 + c
            perms.append(perm)
    arr = np.stack(perms, axis=0)  # (8, 16)
    return arr


SYM_PERM = _build_sym_perm()


def _build_sym_patterns():
    """(NUM_BASE, 8, 6) int64 array: every base pattern mapped through
    all 8 symmetries. sym_patterns[b, s] is the 6 cell indices of the
    s-th oriented variant of base pattern b."""
    out = np.empty((NUM_BASE, NUM_SYM, TUPLE_LEN), dtype=np.int64)
    for b, pat in enumerate(BASE_PATTERNS):
        for s in range(NUM_SYM):
            for j, cell in enumerate(pat):
                out[b, s, j] = SYM_PERM[s][cell]
    return out


SYM_PATTERNS = _build_sym_patterns()


# --- Storage constructors --------------------------------------------
def make_weights(num_stages=1, init_v=0.0):
    """Weight table, shape (num_stages, NUM_BASE, 16**6), float32.
    ~0.5 GB per stage -- see module docstring.

    `init_v` = OPTIMISTIC INITIALIZATION target: every weight is set so
    that the initial value V(any board) == init_v (since V sums NUM_BASE*8
    entries, each entry starts at init_v / (NUM_BASE*8)). init_v=0 gives
    the plain zero start. A positive init_v makes the agent optimistic
    about unseen states, which both accelerates early learning and keeps
    the greedy policy exploring until real returns pull V down -- a
    well-documented win for N-tuple 2048 (Jaskowski 2017). Pick init_v on
    the order of an early-achievable score (e.g. a few x10^4)."""
    w = np.zeros((num_stages, NUM_BASE, INDEX_SIZE), dtype=np.float32)
    if init_v != 0.0:
        w[:] = np.float32(init_v / (NUM_BASE * NUM_SYM))
    return w


def make_tc_accumulators(num_stages=1):
    """TC accumulators (E, A), each same shape/dtype as the weights.
    E = running sum of signed update shares, A = running sum of |shares|.
    The learner reads/writes these via update()."""
    e = np.zeros((num_stages, NUM_BASE, INDEX_SIZE), dtype=np.float32)
    a = np.zeros((num_stages, NUM_BASE, INDEX_SIZE), dtype=np.float32)
    return e, a


# --- njit kernels -----------------------------------------------------
@njit(cache=True)
def feature_index(board, pattern):
    """Pack the 6 cells named by `pattern` into a 24-bit feature index.

    cell value is clamped to TILE_CAP(15); physical 4-bit cells already
    cap there so this is defensive. `board` is a uint64; `pattern` a
    length-6 int array. Returns int64 in [0, 16**6).
    """
    idx = np.int64(0)
    for j in range(pattern.shape[0]):
        c = pattern[j]
        v = np.int64((board >> np.uint64(4 * np.int64(c))) & np.uint64(0xF))
        if v > 15:
            v = np.int64(15)
        idx |= v << np.int64(4 * j)
    return idx


@njit(cache=True)
def evaluate(board, weights, stage, sym_patterns):
    """N-tuple value of `board` at `stage`.

    Sum over all base tuples and all 8 symmetric variants of the weight
    entry each one indexes. `weights` is the full
    (NUM_STAGES, NUM_BASE, 16**6) array; `sym_patterns` the
    (NUM_BASE, 8, 6) index array (pass the module SYM_PATTERNS). Returns
    float64. Sizes are read from the arrays (not hardcoded) so the
    kernels can be warmed up on a tiny dummy weights array.
    """
    total = 0.0
    nbase = sym_patterns.shape[0]
    nsym = sym_patterns.shape[1]
    for b in range(nbase):
        for s in range(nsym):
            idx = feature_index(board, sym_patterns[b, s])
            total += weights[stage, b, idx]
    return total


@njit(cache=True)
def update(board, weights, E, A, stage, sym_patterns, delta, alpha_floor=0.0):
    """Apply a TC (temporal-coherence) update spreading `delta` evenly
    across all NUM_BASE*8 touched weight entries.

    For each touched entry: per-weight learning rate
        alpha = max(|E[entry]| / A[entry], alpha_floor)   (1.0 when A == 0)
    computed from the CURRENT accumulators, then
        w[entry]  += alpha * delta_share
        E[entry]  += delta_share
        A[entry]  += |delta_share|
    with delta_share = delta / (NUM_BASE*8).

    `alpha_floor` (default 0.0 = plain TC) is a LOWER BOUND on the per-
    weight learning rate. Vanilla TC drives alpha -> 0 for any weight
    whose updates keep flipping sign (|E| stays small while A grows), i.e.
    it FREEZES that weight. That is usually desirable (stop chasing spawn
    noise), but it also freezes weights prematurely while the policy is
    still improving -- a cause of the hard mid-training plateau. A small
    floor (e.g. 0.01) keeps every weight minimally plastic so learning can
    keep improving past the plateau, at the cost of a little residual
    noise-chasing.

    This is a concrete, standard TC formulation; the learner may bypass
    this and write weights directly if it wants a different update rule.
    """
    nbase = sym_patterns.shape[0]
    nsym = sym_patterns.shape[1]
    n_entries = nbase * nsym
    delta_share = delta / n_entries
    abs_share = delta_share if delta_share >= 0.0 else -delta_share
    for b in range(nbase):
        for s in range(nsym):
            idx = feature_index(board, sym_patterns[b, s])
            a_val = A[stage, b, idx]
            if a_val > 0.0:
                e_val = E[stage, b, idx]
                abs_e = e_val if e_val >= 0.0 else -e_val
                alpha = abs_e / a_val
                if alpha < alpha_floor:
                    alpha = alpha_floor
            else:
                alpha = 1.0
            weights[stage, b, idx] += np.float32(alpha * delta_share)
            E[stage, b, idx] += np.float32(delta_share)
            A[stage, b, idx] += np.float32(abs_share)


def _warmup():
    """Trigger JIT compilation at import on a tiny dummy weights array so
    the first real call isn't slow. Uses a zero board (every feature
    index is 0, which fits the tiny dummy index dimension) -- real
    weights MUST be full size (INDEX_SIZE) to hold true feature indices.
    """
    try:
        tiny_w = np.zeros((1, NUM_BASE, 16), dtype=np.float32)
        tiny_e = np.zeros((1, NUM_BASE, 16), dtype=np.float32)
        tiny_a = np.zeros((1, NUM_BASE, 16), dtype=np.float32)
        b0 = np.uint64(0)
        feature_index(b0, SYM_PATTERNS[0, 0])
        evaluate(b0, tiny_w, 0, SYM_PATTERNS)
        update(b0, tiny_w, tiny_e, tiny_a, 0, SYM_PATTERNS, 1.0)
    except Exception:
        pass


_warmup()
