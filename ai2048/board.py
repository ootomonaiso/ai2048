"""Pure-Python board operations for the 2048 game loop.

Not performance critical (called once per real turn, not per search
node) so it favors clarity over speed. Shares the precomputed row
tables in tables.py with the numba search core in fastcore.py, so both
layers agree on move semantics.
"""
import random

from .tables import LEFT_RESULT, LEFT_SCORE, RIGHT_RESULT, RIGHT_SCORE, BOARD_MASK, ROW_MASK

CORNERS = (0, 3, 12, 15)


def transpose(board):
    a1 = board & 0xF0F00F0FF0F00F0F
    a2 = board & 0x0000F0F00000F0F0
    a3 = board & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def _move_rows(board, result_table, score_table):
    result = 0
    score = 0
    moved = False
    for i in range(4):
        row = (board >> (16 * i)) & ROW_MASK
        new_row = int(result_table[row])
        if new_row != row:
            moved = True
        score += int(score_table[row])
        result |= new_row << (16 * i)
    return result, score, moved


def move_left(board):
    return _move_rows(board, LEFT_RESULT, LEFT_SCORE)


def move_right(board):
    return _move_rows(board, RIGHT_RESULT, RIGHT_SCORE)


def move_up(board):
    t = transpose(board)
    result, score, moved = _move_rows(t, LEFT_RESULT, LEFT_SCORE)
    return transpose(result), score, moved


def move_down(board):
    t = transpose(board)
    result, score, moved = _move_rows(t, RIGHT_RESULT, RIGHT_SCORE)
    return transpose(result), score, moved


MOVES = (move_left, move_right, move_up, move_down)
MOVE_NAMES = ("Left", "Right", "Up", "Down")


def empty_cells(board):
    return [i for i in range(16) if ((board >> (4 * i)) & 0xF) == 0]


def count_empty(board):
    return sum(1 for i in range(16) if ((board >> (4 * i)) & 0xF) == 0)


def set_cell(board, index, value):
    board &= (~(0xF << (4 * index))) & BOARD_MASK
    board |= (value & 0xF) << (4 * index)
    return board


def get_cell(board, index):
    return (board >> (4 * index)) & 0xF


def spawn_tile(board, rng=random):
    cells = empty_cells(board)
    if not cells:
        return board
    idx = rng.choice(cells)
    value = 1 if rng.random() < 0.9 else 2
    return set_cell(board, idx, value)


def to_grid(board):
    return [[get_cell(board, r * 4 + c) for c in range(4)] for r in range(4)]


def from_grid(grid):
    board = 0
    for r in range(4):
        for c in range(4):
            board = set_cell(board, r * 4 + c, grid[r][c])
    return board


def max_exponent(board):
    return max(get_cell(board, i) for i in range(16))
