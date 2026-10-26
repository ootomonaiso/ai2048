"""Tkinter GUI: watch the Expectimax AI play 2048 with colored tiles."""
import argparse
import random
import time
import tkinter as tk

from ai2048 import board as B
from ai2048.game import Game
from ai2048.fastcore import best_move_fast

TILE_COLORS = {
    0: ("#cdc1b4", "#776e65"),
    1: ("#eee4da", "#776e65"),   # 2
    2: ("#ede0c8", "#776e65"),   # 4
    3: ("#f2b179", "#f9f6f2"),   # 8
    4: ("#f59563", "#f9f6f2"),   # 16
    5: ("#f67c5f", "#f9f6f2"),   # 32
    6: ("#f65e3b", "#f9f6f2"),   # 64
    7: ("#edcf72", "#f9f6f2"),   # 128
    8: ("#edcc61", "#f9f6f2"),   # 256
    9: ("#edc850", "#f9f6f2"),   # 512
    10: ("#edc53f", "#f9f6f2"),  # 1024
    11: ("#edc22e", "#f9f6f2"),  # 2048
}
DEFAULT_COLOR = ("#3c3a32", "#f9f6f2")

BG = "#bbada0"
CELL = 100
GAP = 12
PAD = 12
BOARD_PX = PAD * 2 + CELL * 4 + GAP * 3


class App:
    def __init__(self, root, time_limit=0.2, min_depth=3, max_depth=11, delay_ms=80, seed=None,
                 policy=None, agent_label="heuristic"):
        self.root = root
        self.time_limit = time_limit
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.delay_ms = delay_ms
        self.policy = policy          # if set, a board_int -> move_idx|None callable (trained net)
        self.agent_label = agent_label
        self.running = False
        self.move_count = 0

        self.canvas = tk.Canvas(root, width=BOARD_PX, height=BOARD_PX, bg=BG, highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)

        info = tk.Frame(root)
        info.pack(fill="x", padx=10)
        self.score_var = tk.StringVar()
        self.status_var = tk.StringVar()
        tk.Label(info, textvariable=self.score_var, font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(info, textvariable=self.status_var, font=("Segoe UI", 10)).pack(side="right")

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=10, pady=(0, 10))
        self.toggle_btn = tk.Button(controls, text="Start", width=8, command=self.toggle)
        self.toggle_btn.pack(side="left")
        tk.Button(controls, text="New Game", command=self.new_game).pack(side="left", padx=6)
        tk.Button(controls, text="Step", command=self.single_step).pack(side="left")

        tk.Label(controls, text="Speed").pack(side="left", padx=(16, 2))
        self.speed_scale = tk.Scale(
            controls, from_=0, to=500, orient="horizontal", length=160,
            showvalue=False, command=self.on_speed_change,
        )
        self.speed_scale.set(delay_ms)
        self.speed_scale.pack(side="left")

        self.game = Game(random.Random(seed))
        self.draw()

    def on_speed_change(self, v):
        self.delay_ms = int(float(v))

    def new_game(self):
        self.running = False
        self.toggle_btn.config(text="Start")
        self.game = Game(random.Random())
        self.move_count = 0
        self.status_var.set("")
        self.draw()

    def toggle(self):
        self.running = not self.running
        self.toggle_btn.config(text="Pause" if self.running else "Start")
        if self.running:
            self.root.after(10, self.step)

    def single_step(self):
        self.running = False
        self.toggle_btn.config(text="Start")
        self._do_move()

    def step(self):
        if not self.running:
            return
        self._do_move()
        if self.running:
            self.root.after(self.delay_ms, self.step)

    def _do_move(self):
        if self.game.is_game_over():
            self.running = False
            self.toggle_btn.config(text="Start")
            self.status_var.set("GAME OVER")
            return
        t0 = time.perf_counter()
        if self.policy is not None:
            mv = self.policy(self.game.board)
        else:
            mv = best_move_fast(
                self.game.board, time_limit=self.time_limit,
                min_depth=self.min_depth, max_depth=self.max_depth,
            )
        dt = time.perf_counter() - t0
        if mv is None:
            self.running = False
            self.toggle_btn.config(text="Start")
            self.status_var.set("GAME OVER")
            return
        self.game.step(mv)
        self.move_count += 1
        self.status_var.set(f"move {self.move_count}  think {dt * 1000:.0f}ms  ({B.MOVE_NAMES[mv]})")
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        grid = B.to_grid(self.game.board)
        for r in range(4):
            for c in range(4):
                v = grid[r][c]
                x0 = PAD + c * (CELL + GAP)
                y0 = PAD + r * (CELL + GAP)
                x1, y1 = x0 + CELL, y0 + CELL
                bg, fg = TILE_COLORS.get(v, DEFAULT_COLOR) if v else TILE_COLORS[0]
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=bg, outline="")
                if v:
                    val = 1 << v
                    fsize = 32 if val < 128 else (28 if val < 1024 else 22)
                    self.canvas.create_text(
                        (x0 + x1) // 2, (y0 + y1) // 2, text=str(val), fill=fg,
                        font=("Segoe UI", fsize, "bold"),
                    )
        self.score_var.set(f"Score: {self.game.score}   Max: {self.game.max_tile()}")


def main():
    parser = argparse.ArgumentParser(description="Watch the Expectimax AI play 2048 in a GUI")
    parser.add_argument("--time", type=float, default=0.2, help="time budget per move (s)")
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=11)
    parser.add_argument("--delay", type=int, default=80, help="ms between moves when auto-playing")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto", action="store_true", help="start auto-playing immediately")
    parser.add_argument("--net", default="", help="checkpoint dir of a trained N-tuple net "
                        "(e.g. runs/ms2_5t8). If set, the net plays instead of the hand heuristic.")
    parser.add_argument("--net-depth", type=int, default=1, help="FIXED expectimax depth for the net "
                        "(used only if --net-time is 0): 0=1-ply, 1/2=deeper")
    parser.add_argument("--net-time", type=float, default=0.1, help="per-move time budget (s) for "
                        "ADAPTIVE-depth net search (default): shallow/fast in the open early game, "
                        "automatically deep in the crowded late game (where it's cheap and matters "
                        "most -- directly strengthens the endgame). Set 0 to use fixed --net-depth.")
    args = parser.parse_args()

    policy = None
    label = "heuristic"
    title = "2048 Expectimax AI"
    if args.net:
        from ai2048 import netsearch as NS
        w = NS.load_weights(args.net)
        thr = NS.load_thresholds(args.net)
        if args.net_time > 0:
            policy = NS.make_policy(w, thresholds=thr, time_limit=args.net_time,
                                    max_depth=7, use_tt=True)
            label = f"net:{args.net} adaptive {args.net_time}s"
            title = f"2048 N-tuple AI (adaptive {args.net_time}s/move)"
        else:
            policy = NS.make_policy(w, depth=args.net_depth, thresholds=thr)
            label = f"net:{args.net} d{args.net_depth}"
            title = f"2048 N-tuple AI (depth {args.net_depth})"

    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    app = App(
        root, time_limit=args.time, min_depth=args.min_depth,
        max_depth=args.max_depth, delay_ms=args.delay, seed=args.seed,
        policy=policy, agent_label=label,
    )
    if args.auto:
        app.toggle()
    root.mainloop()


if __name__ == "__main__":
    main()
