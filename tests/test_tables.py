"""Tests for the precomputed row-move tables and board bit-twiddling.

These guard the bit-level board representation, which is the kind of code
that fails silently (a wrong mask just produces subtly wrong boards).
Run with:  python -m pytest tests/  (or plain: python tests/test_tables.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ai2048.tables import (
    LEFT_RESULT, LEFT_SCORE, RIGHT_RESULT, RIGHT_SCORE,
    _unpack_row, _pack_row, MAX_EXPONENT,
)
from ai2048 import board as B


def _tile_sum(cells):
    """Sum of the actual tile VALUES (2**exp) of a row's cells."""
    return sum((1 << c) if c else 0 for c in cells)


def test_pack_unpack_roundtrip():
    for row in (0, 0x1234, 0xFFFF, 0xABCD):
        assert _pack_row(_unpack_row(row)) == row


def test_left_move_conserves_tile_mass_for_all_rows():
    """A slide/merge must never create or destroy tile mass: the sum of
    2**exp over the row is invariant (merging two 2^k makes one 2^(k+1),
    same total). This is exactly what the 32768-overflow bug violated --
    it turned two 32768 tiles into empty cells (mass 65536 -> 0)."""
    for row in range(65536):
        cells = _unpack_row(row)
        before = _tile_sum(cells)
        for table in (LEFT_RESULT, RIGHT_RESULT):
            after = _tile_sum(_unpack_row(int(table[row])))
            assert after == before, (
                f"tile mass changed on row {row:#06x}: {before} -> {after}"
            )


def test_two_32768_do_not_merge():
    """The regression test for the overflow bug: two adjacent 32768 tiles
    (exponent 15) must stay as two 32768 tiles, not vanish."""
    row = MAX_EXPONENT | (MAX_EXPONENT << 4)  # [15, 15, 0, 0]
    res = _unpack_row(int(LEFT_RESULT[row]))
    assert res == [MAX_EXPONENT, MAX_EXPONENT, 0, 0], res
    assert int(LEFT_SCORE[row]) == 0, "no score should be awarded for a blocked merge"


def test_score_matches_merges():
    """[2,2,4,4] -> merging gives a 4 (score 4) and an 8 (score 8)."""
    row = _pack_row([1, 1, 2, 2])
    assert int(LEFT_SCORE[row]) == (1 << 2) + (1 << 3)


def test_transpose_is_involution():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        b = int(rng.integers(0, 1 << 63)) & B.BOARD_MASK
        assert B.transpose(B.transpose(b)) == b


def test_set_get_cell_roundtrip():
    b = 0
    for i in range(16):
        b = B.set_cell(b, i, (i % 15) + 1)
    for i in range(16):
        assert B.get_cell(b, i) == (i % 15) + 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
