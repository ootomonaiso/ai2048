"""numba-JIT self-play + TD-afterstate(+TC) learning kernels.

This is the hot path of training. Everything here is @njit so a whole
batch of self-play games (each playing to the end and applying TD
updates along the way) runs in native code, parallelized across CPU
cores with numba prange + Hogwild-style (unsynchronized) updates to the
shared weight / TC-accumulator arrays. Races are accepted -- this is the
standard way N-tuple 2048 nets are trained in parallel and it converges
fine.

Algorithm: TD-afterstate learning (Szubert & Jaskowski). The value
function V estimates expected FUTURE reward from an *afterstate* (the
board right after the player's move, before the random tile spawn). The
greedy policy picks the move maximizing (immediate reward + V(afterstate)).
For a chosen afterstate `after`, once we see the next afterstate `after2`
(greedily chosen from the spawned board) with its move reward `reward2`,
the TD target for V(after) is `reward2 + V(after2)` (or 0 at terminal),
and the TC `update` nudges V(after) toward it.

RNG: a per-game xorshift64 state (a plain local uint64) so parallel games
never share RNG state -- fully thread-safe under prange.
"""
import numpy as np
from numba import njit, prange

from . import ntuple as NT
from .fastcore import move_left, move_right, move_up, move_down


# --- thread-safe RNG --------------------------------------------------
@njit(cache=True)
def _xorshift(state):
    """xorshift64. state must be nonzero. Returns (random_uint64, new_state)."""
    x = state
    x ^= (x << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(7)
    x ^= (x << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x, x


@njit(cache=True)
def _spawn(board, state):
    """Place one random tile (90% a 2 = exp1, 10% a 4 = exp2) on a random
    empty cell. Returns (new_board, new_state)."""
    # count empties
    n = 0
    for i in range(16):
        if ((board >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:
            n += 1
    if n == 0:
        return board, state
    r, state = _xorshift(state)
    k = np.int64(r % np.uint64(n))  # k-th empty cell
    # pick value
    r2, state = _xorshift(state)
    val = np.uint64(1)
    if (r2 % np.uint64(10)) == np.uint64(0):
        val = np.uint64(2)
    # find the k-th empty cell and set it
    seen = 0
    for i in range(16):
        if ((board >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:
            if seen == k:
                board = board | (val << np.uint64(4 * i))
                break
            seen += 1
    return board, state


@njit(cache=True)
def _init_board(state):
    board = np.uint64(0)
    board, state = _spawn(board, state)
    board, state = _spawn(board, state)
    return board, state


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
    """Multistage selector: stage = how many max-tile-exponent thresholds
    the board's max tile meets or exceeds. An empty `thresholds` array
    means single-stage (always 0). Example thresholds [13,14,15] ->
    stage 0 below 8192, 1 at 8192, 2 at 16384, 3 at >=32768."""
    n = thresholds.shape[0]
    if n == 0:
        return 0
    me = _max_exponent(board)
    s = 0
    for t in range(n):
        if me >= thresholds[t]:
            s += 1
    return s


# --- greedy afterstate selection -------------------------------------
@njit(cache=True)
def _greedy(board, weights, thresholds, sym_patterns):
    """Pick the move maximizing reward + V(afterstate). Each candidate
    afterstate is evaluated at ITS OWN stage (selected by thresholds), so
    multistage value functions are used correctly during play.

    Returns (best_after, best_reward, any_legal, best_value) where
    best_value = best_reward + V_{stage(best_after)}(best_after).
    best_after/best_reward are only meaningful when any_legal is True.
    """
    best_val = -1.0e30
    best_after = np.uint64(0)
    best_reward = 0
    any_legal = False

    nb, sc, mv = move_left(board)
    if mv:
        any_legal = True
        v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym_patterns)
        if v > best_val:
            best_val = v
            best_after = nb
            best_reward = sc
    nb, sc, mv = move_right(board)
    if mv:
        any_legal = True
        v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym_patterns)
        if v > best_val:
            best_val = v
            best_after = nb
            best_reward = sc
    nb, sc, mv = move_up(board)
    if mv:
        any_legal = True
        v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym_patterns)
        if v > best_val:
            best_val = v
            best_after = nb
            best_reward = sc
    nb, sc, mv = move_down(board)
    if mv:
        any_legal = True
        v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym_patterns)
        if v > best_val:
            best_val = v
            best_after = nb
            best_reward = sc

    return best_after, best_reward, any_legal, best_val


@njit(cache=True)
def best_move_net(board, weights, thresholds, sym_patterns):
    """Move index (0=L,1=R,2=U,3=D) maximizing reward + V(afterstate), or
    -1 if no legal move. For 1-ply (search-free) net play / evaluation.
    Multistage-aware (stage selected per afterstate)."""
    best_val = -1.0e30
    best_idx = -1
    nb, sc, mv = move_left(board)
    if mv:
        v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym_patterns)
        if v > best_val:
            best_val = v; best_idx = 0
    nb, sc, mv = move_right(board)
    if mv:
        v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym_patterns)
        if v > best_val:
            best_val = v; best_idx = 1
    nb, sc, mv = move_up(board)
    if mv:
        v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym_patterns)
        if v > best_val:
            best_val = v; best_idx = 2
    nb, sc, mv = move_down(board)
    if mv:
        v = sc + NT.evaluate(nb, weights, _stage_of(nb, thresholds), sym_patterns)
        if v > best_val:
            best_val = v; best_idx = 3
    return best_idx


# --- one self-play game with TD-afterstate(+TC) learning --------------
_LAMBDA_MAXT = 4096  # trajectory buffer per game for the lambda-return path


@njit(cache=True)
def play_and_learn_one(seed, weights, E, A, thresholds, sym_patterns, lr,
                       alpha_floor, start_board, collect_min_exp, lam):
    """Play one game greedily under the current net while applying
    TD-afterstate + TC updates. Multistage: each afterstate is valued and
    updated at its own stage (empty `thresholds` -> single stage). The TD
    target for V_{stage(after)}(after) is reward2 + V_{stage(after2)}(after2)
    and may cross a stage boundary (standard). `seed` must be nonzero.

    RESTART / CAROUSEL SHAPING. If `start_board` is nonzero, the game does
    NOT start from an empty board -- it resumes from `start_board` (a
    previously-encountered afterstate), so self-play spends time in the
    rare late-game region that fresh games almost never reach. This is the
    fix for the hard plateau on 16384/32768: those positions' weights only
    get trained if the learner actually visits them. (start_board=0 = the
    normal empty start.)

    POOL COLLECTION. If `collect_min_exp > 0`, the game reservoir-samples
    ONE afterstate whose max-tile exponent is >= collect_min_exp and
    returns it as `deep` (0 if none qualified), so the driver can grow a
    pool of high-tile states to restart future games from. Uniform over
    all qualifying afterstates in the game (reservoir sampling).

    DELAYED-TC(lambda). `lam`=0 -> plain online TD(0): each afterstate is
    updated immediately by its 1-step error (the original, validated path,
    byte-identical). `lam`>0 -> forward-view TD(lambda) (Jaskowski's
    delayed TC): the whole episode is played with the weights held FIXED
    (so delta_t are consistent), then updates are applied in one backward
    pass where each afterstate's error is the lambda-return error
    Delta_t = delta_t + lam*Delta_{t+1} = sum_k lam^k delta_{t+k}. Mixing
    multi-step returns cuts the single-sample spawn-noise variance that
    makes a high lr diverge under plain TC, which is what lets lr approach
    1.0. The TC accumulators E/A then accumulate these Delta shares (E+=D,
    A+=|D|), exactly as in the paper's Algorithm 4.

    Returns (score, max_exp, deep_afterstate)."""
    state = np.uint64(seed)
    if state == np.uint64(0):
        state = np.uint64(0x9E3779B97F4A7C15)

    if start_board != np.uint64(0):
        # resume from a pooled afterstate
        after = start_board
        board = start_board
        v_after = NT.evaluate(after, weights, _stage_of(after, thresholds), sym_patterns)
        score = np.int64(0)
    else:
        board, state = _init_board(state)
        after, reward, legal, best_val = _greedy(board, weights, thresholds, sym_patterns)
        if not legal:
            return 0, _max_exponent(board), np.uint64(0)
        score = np.int64(reward)
        v_after = best_val - reward  # = V_{stage(after)}(after)

    deep = np.uint64(0)
    seen_high = np.uint64(0)

    if lam == 0.0:
        # ---- plain online TD(0) (original validated path) ----
        while True:
            board, state = _spawn(after, state)
            after2, reward2, legal2, best_val2 = _greedy(board, weights, thresholds, sym_patterns)
            stage_after = _stage_of(after, thresholds)
            if not legal2:
                err = 0.0 - v_after
                NT.update(after, weights, E, A, stage_after, sym_patterns, lr * err, alpha_floor)
                return score, _max_exponent(board), deep
            target = best_val2  # = reward2 + V_{stage(after2)}(after2)
            err = target - v_after
            NT.update(after, weights, E, A, stage_after, sym_patterns, lr * err, alpha_floor)
            score += reward2
            if collect_min_exp > 0 and _max_exponent(after2) >= collect_min_exp:
                seen_high += np.uint64(1)
                r, state = _xorshift(state)
                if (r % seen_high) == np.uint64(0):
                    deep = after2
            after = after2
            v_after = best_val2 - reward2

    # ---- delayed-TC(lambda): buffer the episode, apply lambda-return at end ----
    traj_after = np.empty(_LAMBDA_MAXT, dtype=np.uint64)
    traj_stage = np.empty(_LAMBDA_MAXT, dtype=np.int64)
    traj_delta = np.empty(_LAMBDA_MAXT, dtype=np.float64)
    nt = 0
    while True:
        if nt == _LAMBDA_MAXT:
            # buffer full: flush a (truncated) lambda-return pass, then keep going.
            # Updating mid-episode makes the weights drift, so this is the
            # bounded-horizon "delayed by h" form -- fine for pathologically
            # long games; typical games fit well under the cap.
            acc = 0.0
            for t in range(nt - 1, -1, -1):
                acc = traj_delta[t] + lam * acc
                NT.update(traj_after[t], weights, E, A, traj_stage[t], sym_patterns,
                          lr * acc, alpha_floor)
            nt = 0
        board, state = _spawn(after, state)
        after2, reward2, legal2, best_val2 = _greedy(board, weights, thresholds, sym_patterns)
        stage_after = _stage_of(after, thresholds)
        if not legal2:
            traj_after[nt] = after
            traj_stage[nt] = stage_after
            traj_delta[nt] = 0.0 - v_after
            nt += 1
            break
        traj_after[nt] = after
        traj_stage[nt] = stage_after
        traj_delta[nt] = best_val2 - v_after  # delta_t = reward2 + V(after2) - V(after)
        nt += 1
        score += reward2
        if collect_min_exp > 0 and _max_exponent(after2) >= collect_min_exp:
            seen_high += np.uint64(1)
            r, state = _xorshift(state)
            if (r % seen_high) == np.uint64(0):
                deep = after2
        after = after2
        v_after = best_val2 - reward2

    acc = 0.0
    for t in range(nt - 1, -1, -1):
        acc = traj_delta[t] + lam * acc
        NT.update(traj_after[t], weights, E, A, traj_stage[t], sym_patterns,
                  lr * acc, alpha_floor)
    return score, _max_exponent(board), deep


@njit(cache=True, parallel=True)
def train_batch(seeds, weights, E, A, thresholds, sym_patterns, lr,
                alpha_floor, start_boards, collect_min_exp, lam,
                out_score, out_maxexp, out_deepboard):
    """Play len(seeds) games in parallel (Hogwild updates to the shared
    weights/E/A). Writes per-game (score, max_exponent, deep_afterstate)
    into the output arrays. Each game uses its own RNG seed so parallelism
    is deterministic per seed and thread-safe.

    `alpha_floor` -> TC learning-rate floor (see ntuple.update).
    `start_boards[i]` -> restart afterstate for game i (0 = empty start).
    `collect_min_exp` -> pool-collection tile threshold (0 = no pooling).
    `lam` -> TD(lambda) decay for delayed TC (0 = plain online TD(0))."""
    n = seeds.shape[0]
    for i in prange(n):
        s, mx, deep = play_and_learn_one(
            seeds[i], weights, E, A, thresholds, sym_patterns, lr,
            alpha_floor, start_boards[i], collect_min_exp, lam)
        out_score[i] = s
        out_maxexp[i] = mx
        out_deepboard[i] = deep
