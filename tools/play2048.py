"""2048 speedrun through the cascade: Jev proposes, the solver gate disposes.

An experiment, not a package module. The point is to run the project's own ladder on a
game where the right answer is computable, and measure the split:

1. **Jev routes the strategy.** One Choice at game start: which corner the snake
   heuristic anchors in. Strategy routing is the classification-shaped call the local
   and hosted engines were measured on.
2. **Jev proposes every move.** The board is rendered as text, one typed Choice over
   slide left/right/up/down — the cheap decider, exactly like the browser loop.
3. **The solver gate verifies.** A pure-Python expectimax computes the best move; if
   Jev's pick is not it, the ladder escalates to the solver. Accept/escalate is the
   same shape as the agent loop's verification gate, and the acceptance rate is the
   same measurement: does the cheap decider earn its keep?

The game is headless and deterministic apart from tile spawns, which follow the real
game's rules: a 2 with probability 0.9, a 4 with probability 0.1, on a uniformly random
empty cell.

Usage:
    python3 tools/play2048.py [--depth 2|3] [--seed N] [--max-moves N] [--goal 8192]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.config import load_config  # noqa: E402
from jev_cascade.jev import JevError, build_jev, choice_question  # noqa: E402

GOAL_DEFAULT = 8192
MOVES = ("left", "right", "up", "down")


# ---------------------------------------------------------------------------
# the game

SIZE = 4


def new_board(rng: random.Random) -> list[list[int]]:
    board = [[0] * SIZE for _ in range(SIZE)]
    for _ in range(2):
        spawn(board, rng)
    return board


def spawn(board: list[list[int]], rng: random.Random) -> None:
    empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == 0]
    if not empty:
        return
    r, c = rng.choice(empty)
    board[r][c] = 4 if rng.random() >= 0.9 else 2


def _slide_row(row: list[int]) -> tuple[list[int], int]:
    """One row merged to the left; returns the row and the score gained.

    A tile that was just created by a merge cannot merge again on the same slide —
    the rule that makes [2,2,4,4] become [4,8], not [8].
    """

    tiles = [value for value in row if value]
    out, gained, merged_last = [], 0, False
    for value in tiles:
        if out and out[-1] == value and not merged_last:
            out[-1] = value * 2
            gained += value * 2
            merged_last = True
        else:
            out.append(value)
            merged_last = False
    return out + [0] * (SIZE - len(out)), gained


def _transpose(board):
    return [list(row) for row in zip(*board)]


def _reverse(board):
    return [row[::-1] for row in board]


def move(board: list[list[int]], direction: str) -> tuple[list[list[int]], int, bool]:
    """Apply one slide; returns (board, score gained, whether anything moved)."""

    work = [row[:] for row in board]
    if direction in ("up", "down"):
        work = _transpose(work)
    if direction in ("right", "down"):
        work = _reverse(work)
    result, gained = [], 0
    for row in work:
        slid, points = _slide_row(row)
        result.append(slid)
        gained += points
    if direction in ("right", "down"):
        result = _reverse(result)
    if direction in ("up", "down"):
        result = _transpose(result)
    # Type-consistent comparison: the search passes tuples, the game loop lists,
    # and a list-of-lists never equals a tuple-of-tuples, which once made every
    # no-op slide look legal and cost the solver its endgame.
    moved = _tuples(result) != _tuples(board)
    return result, gained, moved


def can_move(board: list[list[int]]) -> bool:
    return any(move(board, direction)[2] for direction in MOVES)


def max_tile(board: list[list[int]]) -> int:
    return max(max(row) for row in board)


def render(board: list[list[int]]) -> str:
    return "\n".join(" ".join(f"{value or '.':>5}" for value in row) for row in board)


# ---------------------------------------------------------------------------
# the solver: expectimax over a corner-snake heuristic

# Snake weights for the top-left anchor; the other corners are transforms of this grid.
SNAKE_TOP_LEFT = [
    [2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12],
    [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11],
    [2 ** 7, 2 ** 6, 2 ** 5, 2 ** 4],
    [2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3],
]


def snake_weights(corner: str) -> list[list[int]]:
    grid = [row[:] for row in SNAKE_TOP_LEFT]
    if corner in ("top-right", "bottom-right"):
        grid = [row[::-1] for row in grid]
    if corner in ("bottom-left", "bottom-right"):
        grid = grid[::-1]
    return grid


WEIGHTS = {
    # Normalized terms (each ~0..1 relative to the top tile), tuned by --tune.
    # Weight search must run at the depth you play: tuned at depth 2, the weights
    # transferred badly to depth 3 (measured). Hand values are the search seed.
    "order": 10.0,   # snake: big tiles ordered along the corner path
    "empty": 1.0,    # free cells, squared: two empties are four times as good
    "mono": 4.0,     # monotone rows/columns in log space
    "smooth": 0.5,   # neighbours close in log value merge sooner
    "mob": 1.0,      # legal moves right now: trapped boards die
}


def heuristic(board, weights) -> float:
    """Weighted sum of normalized terms. ``board`` is a tuple of tuples here and
    everywhere in the search, so results memoize across moves."""

    flat = [v for row in board for v in row]
    top = max(flat) if any(flat) else 0
    if top == 0:
        return 0.0

    order = sum(
        (board[r][c] / top) * (weights[r][c] / 32768.0)
        for r in range(SIZE)
        for c in range(SIZE)
    ) / 16.0

    empties = sum(1 for v in flat if v == 0)

    mono = 0
    smooth = 0
    for line in list(board) + [tuple(board[r][c] for r in range(SIZE)) for c in range(SIZE)]:
        logs = [v.bit_length() for v in line]
        inc = sum(b - a for a, b in zip(logs, logs[1:]) if b > a)
        dec = sum(a - b for a, b in zip(logs, logs[1:]) if a > b)
        mono += min(inc, dec)
        smooth += sum(abs(a - b) for a, b in zip(logs, logs[1:]) if a and b)

    moves = sum(1 for direction in MOVES if move(board, direction)[2])

    return top * (
        WEIGHTS["order"] * order
        + WEIGHTS["empty"] * (empties ** 2) / 16.0
        - WEIGHTS["mono"] * mono / 24.0
        - WEIGHTS["smooth"] * smooth / 24.0
        + WEIGHTS["mob"] * moves / 4.0
    )


_CACHE: dict = {}


def _tuples(board) -> tuple:
    return tuple(tuple(row) for row in board)


def _expectimax(board: tuple, weights, depth: int, player: bool) -> float:
    """Chance nodes model both tiles at their real probabilities (2 -> 0.9, 4 -> 0.1)
    and sample cells deterministically per board, so the memo cache never serves one
    sample's value to a different sample (the bug that made deeper search worse)."""

    if depth == 0:
        return heuristic(board, weights)
    key = (board, depth, player)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    if player:
        best = float("-inf")
        for direction in MOVES:
            after, _, moved = move(board, direction)
            if not moved:
                continue
            best = max(best, _expectimax(_tuples(after), weights, depth - 1, False))
        value = best if best != float("-inf") else heuristic(board, weights)
    else:
        empties = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == 0]
        if len(empties) > 6:
            empties = random.Random(hash(board)).sample(empties, 6)
        total, work = 0.0, [list(row) for row in board]
        for r, c in empties:
            work[r][c] = 2
            total += 0.9 * _expectimax(_tuples(work), weights, depth - 1, True)
            work[r][c] = 4
            total += 0.1 * _expectimax(_tuples(work), weights, depth - 1, True)
            work[r][c] = 0
        value = total / len(empties) if empties else heuristic(board, weights)
    _CACHE[key] = value
    return value


def best_move(board, weights, depth):
    """The reference move at exactly the depth the caller asks for. (An earlier
    version overrode the caller's depth here, which silently un-did every depth
    experiment - a reminder that a default that fights its caller is a bug.)"""

    scores = {}
    for direction in MOVES:
        after, _, moved = move(board, direction)
        if moved:
            scores[direction] = _expectimax(_tuples(after), weights, depth - 1, False)
    if not scores:
        return None
    return max(scores, key=scores.get)


def board_state(board: list[list[int]], goal: int) -> str:
    tiles = " ".join(sorted((str(v) for row in board for v in row if v), reverse=True)[:8])
    return (f"goal: reach the {goal} tile\nmax tile: {max_tile(board)}\n"
            f"tiles: {tiles}\nboard:\n{render(board)}")


rng_global = random.Random()


def propose_move(jev, board: list[list[int]]) -> tuple[str | None, float]:
    """The cheap decider's pick, or None when the engine is unreachable."""

    options = {f"slide {direction}": f"slide every tile {direction}" for direction in MOVES}
    decision = jev.ask(
        board_state(board, GOAL_DEFAULT),
        {"move": choice_question(options, "Which single slide best sets up the next merge?")},
    )
    choice, confidence = decision.choice("move")
    for direction in MOVES:
        if choice.endswith(direction):
            return direction, confidence
    return "unknown", confidence


def _playout(seed: int, depth: int, weights_override: dict | None = None) -> float:
    """One full game with the current weights; returns log2(max tile) squared,
    so 512 -> 81, 4096 -> 144, 8192 -> 169 - smooth credit for going deeper."""

    p_cache_was = dict(_CACHE)
    _CACHE.clear()
    rng = random.Random(seed)
    board = new_board(rng)
    weights = snake_weights("top-left")
    moves = 0
    while can_move(board) and moves < 4000:
        chosen = best_move(_tuples(board), weights, depth)
        board, _, _ = move(board, chosen)
        spawn(board, rng)
        moves += 1
    _CACHE.clear()
    _CACHE.update(p_cache_was)
    tile = max_tile(board)
    return (tile.bit_length() - 1) ** 2


def tune(rounds: int = 2, depth: int = 2) -> None:
    """Coordinate descent over the eval weights, joint where hand-tuning failed.

    Each candidate is scored by the sum of log-squared final tiles over fixed
    seeds at shallow depth; the winner per coordinate is kept before moving on.
    """

    seeds = (11, 7)
    for round_index in range(rounds):
        for name in WEIGHTS:
            best_value, best_score = WEIGHTS[name], None
            for factor in (0.5, 2.0):
                WEIGHTS[name] = best_value * factor
                score = sum(_playout(seed, depth) for seed in seeds)
                print(f"  round {round_index + 1} {name}={WEIGHTS[name]:<7} score {score}", flush=True)
                if best_score is None or score > best_score:
                    best_value, best_score = WEIGHTS[name], score
            WEIGHTS[name] = best_value
        print(f"round {round_index + 1}: {json.dumps(WEIGHTS)}", flush=True)
    print("final weights:", json.dumps(WEIGHTS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=2, help="expectimax depth (2 fast, 3 stronger)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-moves", type=int, default=4000)
    parser.add_argument("--goal", type=int, default=GOAL_DEFAULT)
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--tune", action="store_true", help="search the eval weights, then exit")
    parser.add_argument("--tune-depth", type=int, default=2,
                        help="depth the weight search plays at; tune at the depth you play")
    parser.add_argument("--tune-rounds", type=int, default=2)
    args = parser.parse_args()

    if args.tune:
        tune(rounds=args.tune_rounds, depth=args.tune_depth)
        return 0

    rng = random.Random(args.seed)
    rng_global.seed(args.seed if args.seed is not None else time.time_ns())
    _CACHE.clear()
    started = time.monotonic()

    jev = build_jev(load_config().jev)
    if getattr(jev, "stub", False):
        print("no decision engine: set $TYPESAFE_API_KEY", file=sys.stderr)
        return 1

    board = new_board(rng)

    # 1) Jev routes the strategy: the snake's anchor corner.
    corners = {f"anchor the {corner} corner": f"the snake orders tiles toward the {corner}"
               for corner in ("top-left", "top-right", "bottom-left", "bottom-right")}
    decision = jev.ask(board_state(board, args.goal), {"corner": choice_question(
        corners, "Which corner should the tile snake be anchored in for this starting board?")})
    corner_choice, corner_conf = decision.choice("corner")
    corner = next((c for c in ("top-left", "top-right", "bottom-left", "bottom-right")
                   if corner_choice.endswith(c)), "top-left")
    weights = snake_weights(corner)
    print(f"strategy: Jev anchored the snake in the {corner} corner (conf={corner_conf:.2f})")

    milestones, decisions, escalations, accepted = [], 0, 0, 0
    engine_fallbacks = 0
    previous_milestone = 0
    dead_reason = ""
    try:
        for move_index in range(1, args.max_moves + 1):
            tile = max_tile(board)
            if tile >= args.goal:
                milestones.append((tile, move_index))
                break
            if not can_move(board):
                dead_reason = "no moves left"
                break
            if jev.total_cost_usd > args.budget_usd:
                dead_reason = f"Jev budget ${args.budget_usd} exhausted"
                break

            # 2) the cheap decider proposes; an unreachable engine degrades the run
            # to solver-only rather than ending it (measured: a 520 outage).
            try:
                pick, confidence = propose_move(jev, board)
                decisions += 1
            except JevError as exc:
                engine_fallbacks += 1
                if engine_fallbacks <= 3:
                    print(f"  engine down ({str(exc)[:60]}); falling back to the solver", flush=True)
                pick, confidence = None, 0.0

            # 3) the gate verifies: solver's best move is the reference, searched
            # adaptively - deeper in the endgame, where runs die and depth is cheap.
            empties_now = sum(1 for row in board for value in row if value == 0)
            depth = 5 if empties_now <= 4 else 4 if empties_now <= 8 else args.depth
            reference = best_move(board, weights, depth)
            if reference is None:
                dead_reason = "solver found no legal move"
                break
            _, _, pick_moved = move(board, pick) if pick in MOVES else (None, 0, False)
            if pick == reference and pick_moved:
                chosen, accepted = pick, accepted + 1
            else:
                chosen, escalations = reference, escalations + 1

            board, _, _ = move(board, chosen)
            spawn(board, rng)

            if move_index % 100 == 0:
                elapsed = time.monotonic() - started
                print(f"  move {move_index:>4}: max {max_tile(board):>5}  "
                      f"({elapsed:5.1f}s, accepted {accepted}/{decisions}, "
                      f"${jev.total_cost_usd:.3f})", flush=True)

            tile = max_tile(board)
            if tile >= previous_milestone * 2 and tile >= 256:
                milestones.append((tile, move_index))
                previous_milestone = tile
                elapsed = time.monotonic() - started
                print(f"  move {move_index:>4}: {tile:>5} tile  "
                      f"({elapsed:5.1f}s, accepted {accepted}/{decisions})")
    except JevError as exc:
        dead_reason = f"the decision engine failed: {exc}"

    elapsed = time.monotonic() - started
    final_tile = max_tile(board)
    print()
    print(render(board))
    print()
    print(f"result    max tile {final_tile} (goal {args.goal}) in {elapsed:.1f}s")
    print(f"moves     {decisions} decided, {accepted} accepted, {escalations} escalated "
          f"({(100 * accepted / decisions if decisions else 0):.0f}% accepted)")
    print(f"engine    {jev.total_cost_usd:.4f} USD in {jev.calls} calls "
          f"({engine_fallbacks} fallbacks to solver-only)")
    print(f"timeline  " + ", ".join(f"{tile}@{move_index}" for tile, move_index in milestones))
    if dead_reason:
        print(f"stopped   {dead_reason}")

    report = {
        "seed": args.seed, "depth": args.depth, "goal": args.goal,
        "corner": corner, "final_tile": final_tile, "elapsed_s": round(elapsed, 1),
        "decisions": decisions, "accepted": accepted, "escalations": escalations,
        "jev_cost_usd": round(jev.total_cost_usd, 4), "jev_calls": jev.calls,
        "engine_fallbacks": engine_fallbacks,
        "milestones": milestones, "stopped": dead_reason or "goal reached",
        "board": board,
    }
    out = Path("runs") / f"2048-{int(time.time())}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report    {out}")
    return 0 if final_tile >= args.goal else 1


if __name__ == "__main__":
    raise SystemExit(main())
