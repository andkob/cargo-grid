from __future__ import annotations
from .state import EnvState

def render_ansi(state: EnvState, width: int, height: int, goal: tuple[int, int]) -> str:
    """
    Convert the environment state into a simple ASCII grid representation.

    Symbols
    A agent position
    P undelivered package
    G goal cell
    # wall cell
    . empty cell

    The agent symbol overrides all others if occupying the same cell.

    Parameters
    state:
        Current environment state.
    width:
        Grid width.
    height:
        Grid height.
    goal:
        Coordinates of the delivery goal.

    Returns
    A multi line string suitable for printing to the terminal.
    """
    ax, ay = state.agent_pos
    gx, gy = goal

    grid = [["." for _ in range(width)] for _ in range(height)]

    for (wx, wy) in state.walls:
        grid[wy][wx] = "#"

    for p in state.packages:
        if not p.delivered:
            px, py = p.pos
            grid[py][px] = "P"

    grid[gy][gx] = "G"
    grid[ay][ax] = "A"

    lines = []
    lines.append(f"step={state.step_count} battery={state.battery} carrying={state.carrying_id}")
    for row in grid:
        lines.append("".join(row))
    return "\n".join(lines)
