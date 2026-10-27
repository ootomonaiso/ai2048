# 2048 AI

A from-scratch 2048 game plus two AIs over a numba-JIT bitboard engine:

1. **Phase 1 — hand-heuristic Expectimax** (`fastcore.py`): depth-limited
   expectimax with a hand-tuned evaluation (empty cells, merges,
   monotonicity, corner). Strong baseline: ~60% reach 8192 at depth 11.
2. **Phase 2 — learned N-tuple network** (`ntuple.py`, `trainkernel.py`,
   `netsearch.py`, `train.py`): the 2048 SOTA approach — an N-tuple value
   function trained by TD-afterstate learning with temporal-coherence
   (TC) adaptive learning rates, via massively parallel self-play. At
   inference the learned value scores the leaves of a shallow expectimax.
   Even an early (200k-game) single-stage net + 1-layer expectimax
   already beats the Phase-1 heuristic (73% vs 60% reach 8192, avg score
   130k vs 96k), and keeps climbing with more training.

## N-tuple network (Phase 2)

- **Network** (`ntuple.py`): a value function = sum of lookup-table
  weights indexed by small tile patterns ("tuples"), summed over all 8
  board symmetries (rotations/reflections share weights → 8x sample
  efficiency). Profile selectable via env `AI2048_NTUPLE`: `5t8`
  (eight 5-tuples, ~100 MB, fits small RAM — default) or `6t8` (eight
  6-tuples, ~1.6 GB, higher ceiling — for the big run when RAM is free).
- **Training** (`train.py` → `trainkernel.py`): TD-afterstate + TC
  learning. Self-play games run in parallel across CPU cores (numba
  `prange`, Hogwild shared-weight updates). Learning rate 0.1 (1.0
  diverges — it chases random-spawn noise). Resumable: weights + TC
  accumulators + stats checkpoint to `runs/<name>/`, auto-resumed on
  restart, so training can be split across many sessions.
- **Inference** (`netsearch.py`): expectimax whose leaves are scored by
  the trained net. `depth=0` = 1-ply afterstate-greedy (fastest);
  `depth>=1` adds chance/search layers (much stronger).
- **Learning-curve levers** (`train.py`, all default-off so existing runs
  resume unchanged). Plain TD(0)+TC learns fast at first, then plateaus
  hard (e.g. a 5-tuple run gained almost nothing from 0.2M → 10M games)
  and barely trains the rare 16384/32768 endgame, because fresh self-play
  almost never reaches it.
  - `--td-lambda L` — **delayed temporal-coherence / TD(λ)** (Jaśkowski
    2018). Replaces the 1-step TD error with the λ-return error applied in
    one backward pass per episode (weights fixed within the episode),
    cutting the single-sample spawn-noise variance that forces the tiny
    `lr 0.1`. **This is the big win: `--td-lambda 0.5 --lr 0.5` learns
    ~5× faster per game** — in a 300k-game 5-tuple A/B it hit avg 122k /
    8192=69% / 16384=3% at 250k games vs the `lr 0.1` baseline's 86k /
    31% / 0%, and led at *every* checkpoint. (`lr 1.0` still diverges even
    with λ, so 0.5 is the sweet spot.) `L=0` = the plain online path.
  - `--restart-prob P` / `--restart-min-exp E` — **restart / carousel
    shaping**: start a fraction `P` of games from a pool of
    previously-seen afterstates whose max tile ≥ `2**E`, so self-play
    actually *practises* the endgame (Jaśkowski 2018 carousel). **Enable
    this LATE** — once a run already reaches 8192 reliably — with a high
    `E` (e.g. 14) so only real endgame states enter the pool. Turning it
    on from scratch with a low `E` *hurts* (it restarts the still-clumsy
    net into states it can't handle, distorting the early distribution;
    measured worse than baseline over 300k games).
  - `--opt-init V` — **optimistic initialization** (Jaśkowski 2017):
    start every board's value at `V` (canonical ≈320000). **Caveat for
    THIS repo:** OI only pays off with a high learning rate that quickly
    burns off the initial optimism. Our plain TC can't run `--lr 1.0`
    stably (it chases spawn-noise variance and diverges — even with OI),
    and at the safe `--lr 0.1` a large `--opt-init` just causes a long
    unrecovered dip (both measured, 300k games, decisively below
    baseline). OI would need a learning rate high enough to burn off the
    optimism fast; delayed TC (now implemented) tames variance at `lr 0.5`
    but `lr 1.0` still diverges even at λ=0.5, so OI stays unvalidated here
    (might need a higher λ). Leave at 0 for now.
  - `--tc-alpha-floor F` — a lower bound on the per-weight TC learning
    rate so vanilla TC can't freeze a weight to 0 permanently (an
    anti-plateau safeguard; this one is ours, not from the papers).

  **Honest status (300k-game A/B):** `--td-lambda 0.5 --lr 0.5` is a clear,
  validated win (above). The others are *not* early-curve wins: optimistic
  init and `lr 1.0` under plain TC both lost badly (`lr 1.0` diverges;
  `--opt-init` at the safe `lr 0.1` just causes a long dip), and restart
  shaping from scratch underperforms — its real payoff is only in the
  endgame regime a 300k run never reaches, so enable it LATE on a long
  run. In short: use delayed TC for the fast curve; keep restart shaping
  for the endgame push; leave opt-init/high-lr off unless combined with
  something that tames the variance.

```
# train (resumable; Ctrl-C any time, re-run to continue):
# recommended fast recipe (delayed TC -- ~5x faster per game than lr-0.1):
python train.py --ckpt runs/s1_5t8 --td-lambda 0.5 --lr 0.5 --batch 512
# plain baseline (the original, if you want the validated-exact dynamics):
python train.py --ckpt runs/s1_5t8 --lr 0.1 --batch 512
# multistage (dedicated endgame weights), warm-started from a single-stage net:
python train.py --ckpt runs/ms2_5t8 --stage-thresholds 14 --init-from runs/s1_5t8
# for the higher ceiling once RAM is free (6-tuples, ~1.6GB+):
AI2048_NTUPLE=6t8 python train.py --ckpt runs/s1_6t8

# Restart/carousel shaping, enabled LATE (after a run already reaches 8192
# reliably) to push the rare 16384/32768 endgame that fresh self-play never
# visits. Use the SAFE lr 0.1 and a HIGH min-exp so only genuine endgame
# states enter the pool. Do NOT turn this on from scratch (see note below).
python train.py --ckpt runs/ms2_5t8 --restart-prob 0.25 --restart-min-exp 14

# evaluate a trained net at various search depths:
python -c "from ai2048.netsearch import load_weights, load_thresholds, make_policy; \
from ai2048.evaluate_agent import evaluate, format_stats; \
w=load_weights('runs/ms2_5t8'); t=load_thresholds('runs/ms2_5t8'); \
print(format_stats(evaluate(make_policy(w, depth=2, thresholds=t), games=20)))"
```

### Final trained agent

`runs/ms2_5t8` — a 2-stage (split at the 16384 tile) warm-started net,
10M self-play games (~16h). Measured strength:

| mode | 8192 | 16384 | avg score |
|---|---|---|---|
| net, 1-ply (instant) | ~100% | few % | ~165k |
| net + depth-2 expectimax | **100%** | ~5-11% | **~180k** |

It builds the 8192 tile essentially every game. Reaching 16384 reliably
(and 32768 = true SOTA) is beyond the 5-tuple network's representational
ceiling — that needs the `6t8` profile (and the RAM for it).

**Adaptive-depth search (the late-game win).** Instead of a fixed search
depth, `netsearch.make_policy(..., time_limit=T)` uses a per-move time
budget with iterative deepening: shallow/instant in the open early game
(huge chance-node branching), and automatically DEEP in the crowded late
game — where few empty cells make deep plies cheap *and* most valuable.
This directly strengthens the endgame (it reaches 16384 where fixed
depth-1 never did), while keeping per-move latency bounded (~`T` seconds).
It's the default net mode in `gui.py`/`play.py`. Watch it play:

```
python gui.py  --net runs/ms2_5t8 --auto            # adaptive 0.1s/move, deep late game
python play.py --net runs/ms2_5t8                   # terminal, same
python gui.py  --net runs/ms2_5t8 --net-time 0.2    # bigger budget = even deeper late game
python gui.py  --net runs/ms2_5t8 --net-time 0 --net-depth 1   # fixed depth instead
```

**Hybrid + genetic algorithm (`hybrid.py`, `ga.py`).** An experiment in
*combining* methods instead of scaling compute: a blended leaf value
(α·net + β·hand-heuristic) in expectimax plus γ·Monte-Carlo rollouts, with
a genetic algorithm evolving the mix (α,β,γ, heuristic sub-weights, search
depth, rollout count/length) by self-play fitness — fitness penalizes
*per-move* compute so the GA prefers efficient combinations over brute
depth. **Finding:** for this net the GA converged back to ≈ pure net — the
heuristic blend and truncated rollouts did not beat the learned value.
The net's endgame blindness isn't patchable by a shallow heuristic or
short rollouts; the real fix remains a stronger value function (6-tuples).
The adaptive-depth search above is the practical late-game improvement
that *did* pan out.

## How it works (Phase-1 engine)

- **Board representation** (`ai2048/tables.py`, `ai2048/board.py`): the
  board is a single 64-bit integer, 16 cells of 4 bits each, storing each
  tile's exponent (0 = empty, n = tile value `2**n`). Every possible
  16-bit row value is precomputed once into a move table (slide+merge
  left/right); up/down moves transpose the board and reuse the same
  tables. This is the same representation/technique used by the
  well-known `nneonneo/2048-ai` (and other strong 2048 bots).
- **Search core** (`ai2048/fastcore.py`): an Expectimax search —
  alternating "max" nodes (the AI picks the best of up to 4 moves) and
  "chance" nodes (the environment randomly places a new 2 or 4 tile) —
  with transposition-table memoization (`numba.typed.Dict`) so
  transpositions reached via different move orders are only evaluated
  once. All of this is `@njit`-compiled by numba for near-C speed
  (no C/C++ toolchain needed). On boards with more than 6 empty cells,
  4-tile spawns are skipped in the chance node (a standard approximation:
  with that many free cells there's no real risk of losing, so it's not
  worth doubling the branching factor to model it).
- **Heuristic** (`evaluate` in `fastcore.py`), used to score leaf boards:
  weighted sum of empty-cell count, count of adjacent equal-value pairs
  (merge potential), a monotonicity score (rewards rows/columns that are
  consistently increasing or decreasing), and a bonus if the largest tile
  sits in a corner.
- **Iterative deepening with adaptive time budget**
  (`best_move_fast` in `fastcore.py`): starts at `min_depth` and keeps
  going one depth deeper as long as the *measured* cost growth between
  the last two depths (with a safety margin) suggests the next depth
  will still fit in the remaining per-move time budget. This matters
  because branching alternates between ~4x (max layer) and
  ~num-empty-cells x (chance layer), so a fixed growth-factor guess is
  not safe — an open board can make one extra depth cost orders of
  magnitude more.

## Usage

```
pip install numba numpy   # already satisfied if numba 0.62+ is installed

python play.py                      # watch the AI play one game, printed to the terminal
python play.py --delay 0.1          # slow it down to watch
python play.py --time 0.3 --max-depth 9   # stronger/slower per move
python play.py --seed 42            # reproducible game

python bench.py -n 20                       # headless stats over 20 games
python bench.py -n 20 --time 0.15 --max-depth 7   # faster/weaker, for a quick smoke run
```

`play.py`/`bench.py` share flags: `--time` (per-move search budget in
seconds), `--min-depth`/`--max-depth` (search depth bounds — note "depth"
here counts half-plies: a max layer + its following chance layer costs 2
units), `--seed`.

The first call into `ai2048.fastcore` pays a one-time numba JIT
compilation cost (a few seconds); after that every move is fast.

## Tuning

`tune.py` runs a coordinate-ascent search over the heuristic weights
`(w_empty, w_merge, w_mono, corner_bonus, w_smooth)` at cheap settings
(shallow depth, short time budget) so many candidates can be tried
quickly, then re-verifies any winning candidate at real shipped settings
before trusting it — tuning at shallow depth can find weights that look
better there but actively hurt deeper search (confirmed empirically: a
promising-looking shallow-tuned candidate dropped the 4096-reach rate
from 60% to 20% once run at real depth). The current defaults in
`fastcore.py` already beat every shallow-tuned candidate tried so far at
real settings, so they're what's shipped.

## Known limits

- Tiles are stored in 4 bits, so the top representable tile is
  `2**15 = 32768` (exponent 15). Two 32768 tiles therefore **do not
  merge** — 32768 is a terminal max tile (the row tables in `tables.py`
  explicitly block the `exp15 + exp15` merge, which would otherwise
  overflow the 4-bit cell to exponent 0 and silently delete both tiles).
  `tests/test_tables.py` guards this (tile-mass conservation over all
  65536 rows). Reaching 65536 is simply out of scope for a 4-bit board.
- The heuristic weights (`W_EMPTY`, `W_MERGE`, `W_MONO`, `CORNER_BONUS` in
  `fastcore.py`) are reasonable hand-picked defaults inspired by publicly
  documented 2048-AI designs, not auto-tuned; tuning them further (e.g.
  via a self-play search over weight combinations) is the main lever
  left for squeezing out more strength.
