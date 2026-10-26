"""Precomputed per-row move tables shared by the pure-Python board logic
and the numba-accelerated search core.

Board representation: a 64-bit integer, 16 cells of 4 bits each, storing
the tile's exponent (0 = empty, n = tile value 2**n). Cells are laid out
row-major: row i occupies bits [16*i, 16*i+16), and within a row, column
c occupies bits [16*i + 4*c, 16*i + 4*c + 4).

For every possible 16-bit row value (65536 of them) we precompute the
result of sliding/merging that row to the left (and to the right), plus
the score gained from merges. Up/down moves are implemented by
transposing the board and reusing the left/right row tables.
"""
import numpy as np


def _unpack_row(row):
    return [(row >> (4 * i)) & 0xF for i in range(4)]


def _pack_row(cells):
    r = 0
    for i, c in enumerate(cells):
        r |= (c & 0xF) << (4 * i)
    return r


def _slide_merge_left(cells):
    tiles = [c for c in cells if c]
    merged = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] + 1
            merged.append(v)
            score += 1 << v
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged.extend([0] * (4 - len(merged)))
    return merged, score


def _build_tables():
    left_result = np.zeros(65536, dtype=np.uint16)
    left_score = np.zeros(65536, dtype=np.int64)
    right_result = np.zeros(65536, dtype=np.uint16)
    right_score = np.zeros(65536, dtype=np.int64)

    for row in range(65536):
        cells = _unpack_row(row)

        new_cells, score = _slide_merge_left(cells)
        left_result[row] = _pack_row(new_cells)
        left_score[row] = score

        rev_cells = cells[::-1]
        new_rev, rscore = _slide_merge_left(rev_cells)
        new_rev.reverse()
        right_result[row] = _pack_row(new_rev)
        right_score[row] = rscore

    return left_result, left_score, right_result, right_score


LEFT_RESULT, LEFT_SCORE, RIGHT_RESULT, RIGHT_SCORE = _build_tables()

BOARD_MASK = 0xFFFFFFFFFFFFFFFF
ROW_MASK = 0xFFFF
