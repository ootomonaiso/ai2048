"""2048 game state: board + score + random tile spawning."""
import random

from . import board as B


class Game:
    def __init__(self, rng=None, seed=None):
        self.rng = rng or random.Random(seed)
        self.board = 0
        self.score = 0
        self.board = B.spawn_tile(self.board, self.rng)
        self.board = B.spawn_tile(self.board, self.rng)

    def legal_moves(self):
        moves = []
        for i, mv in enumerate(B.MOVES):
            _, _, moved = mv(self.board)
            if moved:
                moves.append(i)
        return moves

    def step(self, move_idx):
        """Apply move_idx (0=left,1=right,2=up,3=down). Returns False if illegal."""
        mv = B.MOVES[move_idx]
        new_board, gained, moved = mv(self.board)
        if not moved:
            return False
        self.board = new_board
        self.score += gained
        self.board = B.spawn_tile(self.board, self.rng)
        return True

    def is_game_over(self):
        return len(self.legal_moves()) == 0

    def max_tile(self):
        return 1 << B.max_exponent(self.board)

    def render(self):
        grid = B.to_grid(self.board)
        lines = []
        for row in grid:
            cells = [str((1 << v) if v else ".").rjust(6) for v in row]
            lines.append(" ".join(cells))
        return "\n".join(lines)
