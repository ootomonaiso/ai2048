"""numba-JIT-compiled Expectimax search core for the 2048 AI.

Reuses the row-move tables from tables.py (same board representation:
64-bit int, 16 cells of 4 bits each, cell value = tile exponent, 0 =
empty). All hot-path logic here is @njit(cache=True) so it runs at
near-C speed once compiled; only best_move_fast (the iterative-deepening
driver) runs in plain Python, since it just needs wall-clock timing
between whole-depth search calls.
"""
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict

from .tables import LEFT_RESULT, LEFT_SCORE, RIGHT_RESULT, RIGHT_SCORE

MOVE_NAMES = ("Left", "Right", "Up", "Down")

# Heuristic weights -- tunable. Threaded through evaluate/max_node/
# chance_node/search_root as plain float64 arguments rather than read
# as numba globals, since numba bakes in global constants at compile
# time (a runtime override of these module attributes would silently
# not affect already-jitted code).
W_EMPTY = 270.0
W_MERGE = 700.0
W_MONO = 47.0
CORNER_BONUS = 20000.0
W_SMOOTH = 0.0  # off by default; see tune.py notes
GAME_OVER_PENALTY = 200000.0
DEFAULT_WEIGHTS = (W_EMPTY, W_MERGE, W_MONO, CORNER_BONUS, W_SMOOTH)

KEY_TYPE = types.UniTuple(types.int64, 2)
VALUE_TYPE = types.float64


@njit(cache=True)
def transpose(board):
    a1 = board & np.uint64(0xF0F00F0FF0F00F0F)
    a2 = board & np.uint64(0x0000F0F00000F0F0)
    a3 = board & np.uint64(0x0F0F00000F0F0000)
    a = a1 | (a2 << np.uint64(12)) | (a3 >> np.uint64(12))
    b1 = a & np.uint64(0xFF00FF0000FF00FF)
    b2 = a & np.uint64(0x00FF00FF00000000)
    b3 = a & np.uint64(0x00000000FF00FF00)
    return b1 | (b2 >> np.uint64(24)) | (b3 << np.uint64(24))


@njit(cache=True)
def move_rows(board, result_table, score_table):
    result = np.uint64(0)
    score = 0
    moved = False
    for i in range(4):
        row = int((board >> np.uint64(16 * i)) & np.uint64(0xFFFF))
        new_row = int(result_table[row])
        score += int(score_table[row])
        if new_row != row:
            moved = True
        result |= np.uint64(new_row) << np.uint64(16 * i)
    return result, score, moved


@njit(cache=True)
def move_left(board):
    return move_rows(board, LEFT_RESULT, LEFT_SCORE)


@njit(cache=True)
def move_right(board):
    return move_rows(board, RIGHT_RESULT, RIGHT_SCORE)


@njit(cache=True)
def move_up(board):
    t = transpose(board)
    r, s, m = move_rows(t, LEFT_RESULT, LEFT_SCORE)
    return transpose(r), s, m


@njit(cache=True)
def move_down(board):
    t = transpose(board)
    r, s, m = move_rows(t, RIGHT_RESULT, RIGHT_SCORE)
    return transpose(r), s, m


@njit(cache=True)
def count_empty(board):
    n = 0
    for i in range(16):
        if ((board >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:
            n += 1
    return n


@njit(cache=True)
def max_exponent(board):
    m = 0
    for i in range(16):
        v = int((board >> np.uint64(4 * i)) & np.uint64(0xF))
        if v > m:
            m = v
    return m


@njit(cache=True)
def evaluate(board, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth):
    # Weights are passed in as plain float64 arguments (not read from
    # module globals) so different weight vectors can be tried at the
    # Python call level without forcing numba to recompile anything --
    # numba bakes in global *constants* at compile time, so a runtime
    # `fastcore.W_EMPTY = ...` override would silently no-op.
    cells = np.empty(16, dtype=np.int64)
    for i in range(16):
        cells[i] = int((board >> np.uint64(4 * i)) & np.uint64(0xF))

    empty_count = 0.0
    for i in range(16):
        if cells[i] == 0:
            empty_count += 1.0

    merge_pairs = 0.0
    for r in range(4):
        for c in range(3):
            a = cells[r * 4 + c]
            b = cells[r * 4 + c + 1]
            if a != 0 and a == b:
                merge_pairs += 1.0
    for c in range(4):
        for r in range(3):
            a = cells[r * 4 + c]
            b = cells[(r + 1) * 4 + c]
            if a != 0 and a == b:
                merge_pairs += 1.0

    total_mono = 0.0
    total_smooth = 0.0
    for r in range(4):
        inc = 0.0
        dec = 0.0
        for c in range(3):
            a = cells[r * 4 + c]
            b = cells[r * 4 + c + 1]
            if b > a:
                d = float(b - a)
                inc += d ** 4
            elif a > b:
                d = float(a - b)
                dec += d ** 4
            total_smooth += abs(float(a - b))
        total_mono -= min(inc, dec)
    for c in range(4):
        inc = 0.0
        dec = 0.0
        for r in range(3):
            a = cells[r * 4 + c]
            b = cells[(r + 1) * 4 + c]
            if b > a:
                d = float(b - a)
                inc += d ** 4
            elif a > b:
                d = float(a - b)
                dec += d ** 4
            total_smooth += abs(float(a - b))
        total_mono -= min(inc, dec)

    mx = 0
    for i in range(16):
        if cells[i] > mx:
            mx = cells[i]
    corner_bonus = 0.0
    if cells[0] == mx or cells[3] == mx or cells[12] == mx or cells[15] == mx:
        corner_bonus = corner_bonus_w

    return (w_empty * empty_count + w_merge * merge_pairs + w_mono * total_mono
            + corner_bonus - w_smooth * total_smooth)


@njit
def max_node(board, depth, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth):
    if depth <= 0:
        return evaluate(board, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)

    key = (np.int64(board), np.int64(depth))
    if key in max_cache:
        return max_cache[key]

    best = -1.0e18
    moved_any = False

    nb, sc, mv = move_left(board)
    if mv:
        moved_any = True
        v = chance_node(nb, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
        if v > best:
            best = v
    nb, sc, mv = move_right(board)
    if mv:
        moved_any = True
        v = chance_node(nb, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
        if v > best:
            best = v
    nb, sc, mv = move_up(board)
    if mv:
        moved_any = True
        v = chance_node(nb, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
        if v > best:
            best = v
    nb, sc, mv = move_down(board)
    if mv:
        moved_any = True
        v = chance_node(nb, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
        if v > best:
            best = v

    if not moved_any:
        best = evaluate(board, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth) - GAME_OVER_PENALTY

    max_cache[key] = best
    return best


@njit
def chance_node(board, depth, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth):
    if depth <= 0:
        return evaluate(board, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)

    key = (np.int64(board), np.int64(depth))
    if key in chance_cache:
        return chance_cache[key]

    n = count_empty(board)
    if n == 0:
        return evaluate(board, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)

    consider_four = n <= 6
    total = 0.0
    inv_n = 1.0 / n

    for i in range(16):
        cell_val = (board >> np.uint64(4 * i)) & np.uint64(0xF)
        if cell_val != 0:
            continue
        b2 = board | (np.uint64(1) << np.uint64(4 * i))
        if consider_four:
            total += 0.9 * inv_n * max_node(b2, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
            b4 = board | (np.uint64(2) << np.uint64(4 * i))
            total += 0.1 * inv_n * max_node(b4, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
        else:
            total += inv_n * max_node(b2, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)

    chance_cache[key] = total
    return total


@njit
def search_root(board, depth, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth):
    max_cache = Dict.empty(key_type=KEY_TYPE, value_type=VALUE_TYPE)
    chance_cache = Dict.empty(key_type=KEY_TYPE, value_type=VALUE_TYPE)

    scores = np.full(4, -np.inf, dtype=np.float64)

    nb, sc, mv = move_left(board)
    if mv:
        scores[0] = chance_node(nb, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
    nb, sc, mv = move_right(board)
    if mv:
        scores[1] = chance_node(nb, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
    nb, sc, mv = move_up(board)
    if mv:
        scores[2] = chance_node(nb, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
    nb, sc, mv = move_down(board)
    if mv:
        scores[3] = chance_node(nb, depth - 1, max_cache, chance_cache, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)

    return scores


def best_move_fast(board, time_limit=0.3, min_depth=3, max_depth=9, weights=None):
    """Iterative-deepening driver (plain Python -- only does wall-clock
    bookkeeping between whole-depth njit search_root calls).

    Branching alternates between a "max" layer (fans out by up to 4 legal
    moves) and a "chance" layer (fans out by up to ~num_empty_cells tile
    placements), so the cost growth from one depth to the next is highly
    non-constant -- it can be a ~4x jump or a ~14x+ jump depending on
    which kind of layer was just added. A fixed growth-factor guess is
    not safe (an early open board can make a single extra depth cost
    orders of magnitude more). Instead we track the actual observed
    elapsed-time ratio between the last two completed depths and use
    that (with a safety margin) to decide whether attempting the next
    depth could blow the time budget, before we start it.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    w_empty, w_merge, w_mono, corner_bonus_w, w_smooth = weights

    board_u = np.uint64(board)
    t_start = time.perf_counter()

    DEFAULT_GROWTH_GUESS = 25.0  # conservative guess before we have 2 data points
    MIN_GROWTH_FLOOR = 5.0       # never trust a growth estimate lower than this --
                                 # a single lucky (cache-heavy) transition must not
                                 # make the next, possibly-explosive, transition look safe
    SAFETY_MARGIN = 1.5
    MIN_REMAINING = 0.002  # don't even attempt a depth with near-zero budget left
    HALF_BUDGET_STOP = 0.5  # if the last depth alone ate this fraction of the budget, stop

    # Always complete the min_depth search FIRST, unconditionally, before
    # any budget gate. Otherwise a tight/misconfigured budget (time_limit
    # near 0, or min_depth > max_depth) would exit the loop having run no
    # search at all and return None -- which callers read as "no legal
    # move" and resign on a perfectly playable board. (netsearch's timed
    # driver guarantees its min_depth pass the same way.)
    t0 = time.perf_counter()
    best_scores = search_root(board_u, min_depth, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
    elapsed_history = [time.perf_counter() - t0]
    depth = min_depth + 1

    while depth <= max_depth:
        remaining = time_limit - (time.perf_counter() - t_start)
        if remaining < MIN_REMAINING:
            break

        if elapsed_history and elapsed_history[-1] > time_limit * HALF_BUDGET_STOP:
            break

        if len(elapsed_history) >= 2:
            ratios = [
                elapsed_history[i] / elapsed_history[i - 1]
                for i in range(1, len(elapsed_history))
                if elapsed_history[i - 1] > 0.0
            ]
            growth = max(ratios) if ratios else DEFAULT_GROWTH_GUESS
            growth = max(growth, MIN_GROWTH_FLOOR)
        else:
            growth = DEFAULT_GROWTH_GUESS

        if elapsed_history:
            predicted_next = elapsed_history[-1] * growth * SAFETY_MARGIN
            if predicted_next > remaining:
                break

        t0 = time.perf_counter()
        scores = search_root(board_u, depth, w_empty, w_merge, w_mono, corner_bonus_w, w_smooth)
        this_elapsed = time.perf_counter() - t0
        elapsed_history.append(this_elapsed)
        best_scores = scores
        depth += 1

        if time.perf_counter() - t_start > time_limit:
            break

    if best_scores is None:
        return None
    arr = np.asarray(best_scores)
    if np.all(np.isneginf(arr)):
        return None
    return int(np.argmax(arr))


def _warmup():
    try:
        b = np.uint64(0)
        move_left(b)
        move_right(b)
        move_up(b)
        move_down(b)
        evaluate(b, W_EMPTY, W_MERGE, W_MONO, CORNER_BONUS, W_SMOOTH)
        search_root(b, 1, W_EMPTY, W_MERGE, W_MONO, CORNER_BONUS, W_SMOOTH)
    except Exception:
        pass


_warmup()
