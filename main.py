from collections import deque
import matplotlib
from typing import List, Tuple, Set
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

COLORS = ['red', 'blue', 'green', 'orange', 'cyan', 'magenta']


def find_shortest_path(placed_walls, start, end, blocks=()):
    """Find shortest path between start and end avoiding walls"""
    if start == end:
        return [start]

    reachable_from_start = {start}
    new_children = {start}
    parent = {start: None}

    while new_children:
        last_children = new_children.copy()
        new_children = set()

        for x, y in last_children:
            for n_x, n_y in ((x, y + 1), (x + 1, y), (x - 1, y), (x, y - 1)):
                if (0 <= n_x < rows and 0 <= n_y < cols and
                    not (((n_x, n_y), (x, y)) in placed_walls or ((x, y), (n_x, n_y)) in placed_walls) and
                    (n_x, n_y) not in reachable_from_start) and not ((n_x, n_y) in blocks):

                    parent[(n_x, n_y)] = (x, y)
                    reachable_from_start.add((n_x, n_y))
                    new_children.add((n_x, n_y))

                    if (n_x, n_y) == end:
                        # Reconstruct path
                        path = []
                        current = end
                        while current is not None:
                            path.append(current)
                            current = parent[current]
                        return path[::-1]  # Reverse to get start->end

    raise Exception('No legal path found.')


def is_connected_with_wall(placed_walls, wall):
    """Check if grid remains connected when this wall is added"""
    # Add the new wall to existing walls
    test_walls = placed_walls | {wall}

    v1 = wall[0]
    v2 = wall[1]
    reachable_from_v1 = {v1}
    new_children = {v1}
    while True:
        last_children = new_children.copy()
        new_children = set()
        for x, y in sorted(list(last_children), key=lambda z: (z[0] - v2[0]) ** 2 + (z[1] - v2[1]) ** 2):
            for n_x, n_y in ((x, y + 1), (x + 1, y), (x - 1, y), (x, y - 1)):
                if 0 <= n_x < rows and 0 <= n_y < cols and \
                        (not (((n_x, n_y), (x, y)) in test_walls or ((x, y), (n_x, n_y)) in test_walls)) \
                        and ((n_x, n_y) not in reachable_from_v1):
                    if (n_x, n_y) == v2:
                        return True
                    reachable_from_v1 = reachable_from_v1 | {(n_x, n_y)}
                    new_children = new_children | {(n_x, n_y)}
        if len(new_children) == 0:
            return False


def generate_connected_random_walls(rows, cols, n, seed=None):
    """
    Generate a list of wall segments that keeps the grid connected.
    Each wall is a pair of adjacent cells.
    """
    if seed is not None:
        random.seed(seed)

    if n > (rows - 1) * (cols - 1):
        raise ValueError(
            f"Cannot place {n} walls without disconnecting the maze. Maximum number of placed walls is {(rows - 1) * (cols - 1)}")

    all_possible_walls = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                all_possible_walls.append(((r, c), (r + 1, c)))
            if c + 1 < cols:
                all_possible_walls.append(((r, c), (r, c + 1)))

    placed_walls = set()
    walls_placed = 0
    random.shuffle(all_possible_walls)

    for wall in all_possible_walls:
        if is_connected_with_wall(placed_walls, wall):
            placed_walls.add(wall)
            walls_placed += 1
            print(f'Walls placed: {walls_placed}')
            # draw_maze(placed_walls)
        if walls_placed >= n:
            return list(placed_walls)


def generate_wall_dict(rows, cols, wall_segments):
    """
    Generate a wall dictionary for a (rows x cols) maze.
    Each wall is assigned only once: top or right of a cell.
    Input: wall_segments = list of ((r1, c1), (r2, c2)) pairs
    Output: dict mapping (r, c) to {'top': bool, 'right': bool}
    """
    wall_dict = {}

    # Initialize empty wall flags
    for r in range(rows):
        for c in range(cols):
            wall_dict[(r, c)] = {'top': False, 'right': False}

    for (r1, c1), (r2, c2) in wall_segments:
        # Order the pair so that (cell, direction) is uniquely determined
        if (r1, c1) > (r2, c2):
            r1, c1, r2, c2 = r2, c2, r1, c1

        # Vertical wall (between top/bottom)
        if c1 == c2 and abs(r1 - r2) == 1:
            bottom_cell = max(r1, r2)
            wall_dict[(bottom_cell, c1)]['top'] = True

        # Horizontal wall (between left/right)
        elif r1 == r2 and abs(c1 - c2) == 1:
            left_cell = min(c1, c2)
            wall_dict[(r1, left_cell)]['right'] = True

        else:
            raise ValueError(f"Invalid wall between non-adjacent cells: {(r1, c1)} - {(r2, c2)}")

    return wall_dict


def draw_maze(walls, agents=(), paths=(), goals=(), blocks=(),
              colors=COLORS, crush=(), wall_color='black', walls2=(), wall_colors2='black', cups=(), berries=()):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    # Add transparent gridlines
    for i in range(rows + 1):
        ax.plot([0, cols], [i, i], color='gray', linewidth=2, alpha=0.3)
    for j in range(cols + 1):
        ax.plot([j, j], [0, rows], color='gray', linewidth=2, alpha=0.3)

    # Outer border — match thickness to inner walls

    ax.plot([0, cols], [0, 0], color='black', linewidth=wall_thickness * 2)  # bottom
    ax.plot([0, 0], [0, rows], color='black', linewidth=wall_thickness * 2)  # left
    ax.plot([0, cols], [rows, rows], color='black', linewidth=wall_thickness * 2)  # top
    ax.plot([cols, cols], [0, rows], color='black', linewidth=wall_thickness * 2)  # right

    # Draw each wall

    # Draw walls
    walls_colors = [wall_color, wall_colors2]
    for i, wall_set in enumerate([walls, walls2]):
        wall_dict = generate_wall_dict(rows, cols, wall_set)
        for (r, c), wall in wall_dict.items():
            wall_col = walls_colors[i]
            x, y = c, rows - r - 1
            if wall.get('top'):
                ax.plot([x, x + 1], [y + 1, y + 1], color=wall_col, linewidth=wall_thickness)
            if wall.get('bottom'):
                ax.plot([x, x + 1], [y, y], color=wall_col, linewidth=wall_thickness)
            if wall.get('left'):
                ax.plot([x, x], [y, y + 1], color=wall_col, linewidth=wall_thickness)
            if wall.get('right'):
                ax.plot([x + 1, x + 1], [y, y + 1], color=wall_col, linewidth=wall_thickness)

    for (r, c) in blocks:
        x, y = c, rows - r - 1
        rect = plt.Rectangle((x, y), 1, 1, color='dimgray')
        ax.add_patch(rect)

    for i, path in enumerate(paths):
        for j in range(len(path) - 1):
            (x1, y1), (x2, y2) = path[j], path[j + 1]
            ax.plot([y1 + 0.5, y2 + 0.5], [rows - x1 - 0.5, rows - x2 - 0.5],
                    color=colors[i], linewidth=3, alpha=0.7, zorder=1)

    # Draw agent
    for i, agent in enumerate(agents):
        circle = plt.Circle((agent[1] + 0.5, rows - agent[0] - 0.5), 0.3, zorder=2)
        circle.set_color(colors[i])
        ax.add_patch(circle, )

    for i, goal in enumerate(goals):
        ax.plot(goal[1] + 0.5, rows - goal[0] - 0.5, '*', markersize=25, color=colors[i], zorder=3)

    for i, coord in enumerate(crush):
        x, y = coord
        rays = 12
        cx, cy = x + 0.5, rows - y - 0.5

        radius = 0.2
        white_marker = plt.Circle((cx, cy), radius * 1.2, color='white', alpha=1, zorder=4)
        ax.add_patch(white_marker)

        angles = np.linspace(0, 2 * np.pi, rays, endpoint=False)

        for j, angle in enumerate(angles):
            x0 = cx + np.cos(angle) * 0  # inner point
            y0 = cy + np.sin(angle) * 0
            rads = [1, 0.7, 0.9, 0.7]
            radius_mod = rads[j % len(rads)]
            x1 = cx + np.cos(angle) * radius * radius_mod  # outer point
            y1 = cy + np.sin(angle) * radius * radius_mod
            ax.plot([x0, x1], [y0, y1], color='orange', linewidth=3, zorder=5)

    for cup in cups:
        x, y = cup

        # Calculate coordinates for the cup in the upper-right corner of the square at (x, y)
        cup_x = x + 0.8
        cup_y = rows - y - 0.2

        # Draw the cup's bottom line
        ax.plot([cup_x - 0.1,  + 0.1], [cup_y, cup_y], color='black', linewidth=1)

        # Draw the cup's left side
        ax.plot([cup_x - 0.1,  - 0.1], [cup_y, cup_y + 0.2], color='black', linewidth=1)

        # Draw the cup's right side
        ax.plot([cup_x + 0.1,  + 0.1], [cup_y, cup_y + 0.2], color='black', linewidth=1)

    for berry in berries:
        x, y = berry
        c_x = x + 0.8
        c_y = rows - y - 0.1
        circle_radius = 0.05
        black_circle = plt.Circle((c_x, c_y), circle_radius, color='black', zorder=6)

        # Add the circle to the axes
        ax.add_patch(black_circle)

    # Hide axes
    ax.axis('off')
    plt.show()


def pick_ids(lst, ids):
    if len(lst) == 0:
        return []
    try:
        return [lst[i] for i in ids]
    except IndexError as e:
        raise IndexError(
            f"One or more indices in {ids} are out of bounds for list of length {len(lst)}."
        ) from e


def generate_block_grid(rows: int, cols: int,
                        block_width: int, block_height: int,
                        spacing: int = 2) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Generate a grid of blocked coordinates and intersections.

    Parameters:
        rows (int): Total number of rows in the map.
        cols (int): Total number of columns in the map.
        block_width (int): Width of each rectangular block.
        block_height (int): Height of each rectangular block.
        spacing (int): Spacing (in cells) between blocks.

    Returns:
        Tuple:
            - List of blocked (row, col) coordinates.
            - List of intersection (row, col) coordinates: not in same row or column as any blocked cell.
    """
    blocked: List[Tuple[int, int]] = []
    blocked_rows: Set[int] = set()
    blocked_cols: Set[int] = set()
    if block_width == 0 or block_height == 0:
        return [], []
    for r_start in range(0, rows, block_height + spacing):
        for c_start in range(0, cols, block_width + spacing):
            for dr in range(block_height):
                for dc in range(block_width):
                    r = r_start + dr
                    c = c_start + dc
                    if r < rows and c < cols:
                        blocked.append((r, c))
                        blocked_rows.add(r)
                        blocked_cols.add(c)

    intersections = [(r, c) for r in range(rows)
                     if r not in blocked_rows
                     for c in range(cols)
                     if c not in blocked_cols]

    return blocked, intersections


def generate_goals_for_agents(rows: int, cols: int,
                              agents: List[Tuple[int, int]],
                              blocked: List[Tuple[int, int]],
                              intersections: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    For each agent, generate a goal on the same road and in the correct one-way direction.
    Assumes spacing is 2 and road width is 2.
    Assumes spacing is 2 and road width is 2.
    """
    blocked_set: Set[Tuple[int, int]] = set(blocked)
    intersections_set: Set[Tuple[int, int]] = set(intersections)
    blocked_rows = {r for r, _ in blocked}
    blocked_cols = {c for _, c in blocked}
    goals = []

    for r, c in agents:
        is_horizontal = r not in blocked_rows
        is_vertical = c not in blocked_cols

        if is_horizontal and not is_vertical:
            lane_pos = r % 4  # spacing=2, block height+spacing=4
            right_side = lane_pos < 1
            direction = 1 if right_side else -1

            candidates = [(r, cc) for cc in range(cols)
                          if (r, cc) not in blocked_set
                          and (r, cc) not in intersections_set
                          and cc != c
                          and (cc - c) * direction > 0]

        elif is_vertical and not is_horizontal:
            lane_pos = c % 4
            down_side = lane_pos < 1
            direction = 1 if down_side else -1

            candidates = [(rr, c) for rr in range(rows)
                          if (rr, c) not in blocked_set
                          and (rr, c) not in intersections_set
                          and rr != r
                          and (rr - r) * direction > 0]

        else:
            raise ValueError(f"Agent at ({r}, {c}) is not on a clean horizontal or vertical road.")

        if not candidates:
            raise ValueError(f"No valid goal found for agent at ({r}, {c})")

        goals.append(random.choice(candidates))

    return goals


wall_thickness = 4
SEED = 3
N = 64

walls = [] #[((0, 0), (0, 1)), ((1, 0), (1, 1)), ((2, 0), (2, 1))]  # generate_connected_random_walls(rows, cols, N, SEED)

config1 = [6, 6, 2, 2, 4, [
    (5, 3),  # Left arrow start (red agent)
    (2, 5),  # Right arrow start (red agent)
    (0, 2),  # Top arrow start (red agent)
    (3, 0)  # Bottom arrow start (red agent)
], [
               (0, 3),  # Left arrow points right to goal
               (2, 0),  # Right arrow points left to goal
               (5, 2),  # Top arrow points down to goal
               (3, 5)  # Bottom arrow points up to goal
           ]]

config2 = [
    3, 2, 0, 0, 2, [(0, 0), (2, 1)], [(2, 0), (0, 1)]
]

config3 = [
    3, 3, 1, 1, 2, ((1, 1), (1, 0)), ((0, 1), (1, 0))
]


config4 = [
    2, 2, 0, 0, 2, ((0, 0), (1, 0)), ()
]
rows, cols, block_height, block_width, num_agents, agents, goals = config4
# agents = [agents[0]]
# goals = [goals[0]]
# COLORS = COLORS[0]
walls = [((1, 0), (1, 1))]
blocks, intersections = generate_block_grid(rows, cols, block_height, block_width, spacing=1)
# goals = generate_goals_for_agents(rows, cols, agents, blocks, intersections)
#full_paths = [find_shortest_path(walls, agents[i], goals[i], blocks) for i in range(num_agents)]
#paths = [full_paths[i] for i in range(num_agents)]
paths = [(), ()]
agents_ids = [i for i in range(num_agents)]

#crushes = ((1, 1),)

agents, paths, goals, colors = pick_ids(agents, agents_ids), pick_ids(paths, agents_ids), \
                               pick_ids(goals, agents_ids), pick_ids(COLORS, agents_ids)

draw_maze(walls, agents, paths, goals, blocks, colors, cups=((0, 0), (1, 0)), berries=((1, 0),))
