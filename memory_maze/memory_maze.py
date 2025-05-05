from numba import jit
import numpy as np
import gym
from gym import spaces
from skimage.morphology import flood_fill
import cv2
from tqdm import tqdm
import random
import numba


# enum for environment tile types
import enum
class TileType(enum.Enum):
    EMPTY = 0
    OBSTACLE = 1
    FOOD_SOURCE = 2
    FOOD = 3
    AGENT = 4

class Actions(enum.Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4
    START = 5

valid_move_tiles = {TileType.EMPTY.value, TileType.FOOD_SOURCE.value, TileType.FOOD.value}

class FoodSources:
    def __init__(self, maze_grid: np.ndarray, food_source_count: int, food_energy: float, food_generate_time_average: float, food_generate_time_sdev: float):
        self.food_source_count = food_source_count
        self.food_energy = food_energy
        self.food_generate_time_average = food_generate_time_average
        self.food_generate_time_sdev = food_generate_time_sdev

        # the tuples contain (y, x, time_to_generate_food, food_present)
        self.food_sources = np.zeros((food_source_count, 4), np.int32)
        self.add_food_sources(maze_grid)

    def add_food_sources(self, maze_grid: np.ndarray):
        empty_tile_coordinates = np.argwhere(maze_grid == TileType.EMPTY.value)
        empty_tile_coordinates = empty_tile_coordinates[np.random.choice(empty_tile_coordinates.shape[0], self.food_source_count, replace=False)]
        for i, (y, x) in enumerate(empty_tile_coordinates):
            self.food_sources[i] = (y, x, 0, 0)

    def render(self, maze_grid: np.ndarray):
        for y, x, time_to_generate_food, food_present in self.food_sources:
            if food_present:
                maze_grid[y, x] = TileType.FOOD.value
            else:
                maze_grid[y, x] = TileType.FOOD_SOURCE.value
        return maze_grid

@numba.jit()
def food_step(agent_y: int, agen_x: int, food_sources: np.ndarray, food_energy: float, food_generate_time_average: float, food_generate_time_sdev: float) -> float:
    output_food_energy = 0
    for i in range(food_sources.shape[0]):
        y, x, time_to_generate_food, food_present = food_sources[i]
        if agent_y == y and agen_x == x:
            if food_present:
                output_food_energy += food_energy
                food_sources[i, 2] = 1 + max(int(0.5 + random.gauss(food_generate_time_average, food_generate_time_sdev)), 0)
                food_sources[i, 3] = 0
        elif time_to_generate_food == 0:
            food_sources[i, 2] = -1
            food_sources[i, 3] = 1
        elif time_to_generate_food > 0:
            food_sources[i, 2] = time_to_generate_food-1
            food_sources[i, 3] = 0
    return output_food_energy


@numba.jit()
def add_obstacles(grid: np.ndarray, number_of_obstacles: int) -> np.ndarray:
    """
    Add obstacles to a maze while keeping it fully connected by trial removals and BFS checks.
    Numba-compatible: uses only arrays and loops (no Python dicts/tuples).

    Parameters:
    - grid: 2D np.ndarray of dtype np.uint8, with 0=EMPTY, 1=OBSTACLE. Border must be obstacles.
    - number_of_obstacles: how many obstacles to add; must be less than total empties.

    Returns:
    - grid (modified in-place) with added obstacles.
    """
    h, w = grid.shape
    total_cells = h * w
    # Build array of empty cell flat indices
    empty_ids = np.empty(total_cells, np.int32)
    count_empty = 0
    for idx in range(total_cells):
        if grid[idx // w, idx % w] == 0:
            empty_ids[count_empty] = idx
            count_empty += 1
    # Cannot add more obstacles than empties minus one
    max_obstacles = count_empty - 1
    if number_of_obstacles > max_obstacles:
        number_of_obstacles = max_obstacles

    # BFS support arrays
    visited = np.zeros(total_cells, np.uint8)
    queue = np.empty(total_cells, np.int32)

    obstacles_added = 0
    # Try to add each obstacle
    for _ in range(number_of_obstacles):
        added = False
        # Try at most count_empty attempts for this obstacle
        for _ in range(count_empty):
            # Pick a random empty cell
            pick = np.random.randint(0, count_empty)
            cell = empty_ids[pick]
            r = cell // w
            c = cell % w
            # Tentatively place obstacle
            grid[r, c] = 1
            # Find a start for BFS (any remaining empty)
            start = -1
            for j in range(count_empty):
                if j == pick:
                    continue
                nid = empty_ids[j]
                rr = nid // w
                cc = nid % w
                if grid[rr, cc] == 0:
                    start = nid
                    break
            if start < 0:
                # no empties left
                grid[r, c] = 0
                continue
            # BFS to count reachable empties
            # reset visited
            for i in range(total_cells):
                visited[i] = 0
            head = 0
            tail = 0
            visited[start] = 1
            queue[tail] = start
            tail += 1
            reach = 1
            while head < tail:
                cur = queue[head]; head += 1
                cr = cur // w; cc = cur % w
                # neighbors offsets encoded inline
                # up
                nr = cr - 1; nc = cc
                nid = cur - w
                if grid[nr, nc] == 0 and visited[nid] == 0:
                    visited[nid] = 1
                    queue[tail] = nid; tail += 1; reach += 1
                # down
                nr = cr + 1; nc = cc
                nid = cur + w
                if grid[nr, nc] == 0 and visited[nid] == 0:
                    visited[nid] = 1
                    queue[tail] = nid; tail += 1; reach += 1
                # left
                nr = cr; nc = cc - 1
                nid = cur - 1
                if grid[nr, nc] == 0 and visited[nid] == 0:
                    visited[nid] = 1
                    queue[tail] = nid; tail += 1; reach += 1
                # right
                nr = cr; nc = cc + 1
                nid = cur + 1
                if grid[nr, nc] == 0 and visited[nid] == 0:
                    visited[nid] = 1
                    queue[tail] = nid; tail += 1; reach += 1
            # Check connectivity: reachable empties must equal remaining empties
            if reach == count_empty - 1:
                # commit removal
                # remove this idx from empty_ids
                empty_ids[pick] = empty_ids[count_empty - 1]
                count_empty -= 1
                obstacles_added += 1
                added = True
                break
            else:
                # revert
                grid[r, c] = 0
        if not added:
            # no valid removal found, stop early
            break
    return grid


@jit()
def add_obstacles_python(grid: np.ndarray, number_of_obstacles: int) -> np.ndarray:
    """
    Add obstacles to a maze while keeping it fully connected by building a random spanning tree
    over empty cells (via randomized Prim) and then pruning random leaves.

    Parameters:
    - grid: 2D np.ndarray of dtype np.uint8, with 0=EMPTY, 1=OBSTACLE. Border must be obstacles.
    - number_of_obstacles: how many obstacles to add; must be less than total empties.

    Returns:
    - grid (modified in-place) with added obstacles.

    Raises:
    - ValueError if number_of_obstacles >= total empty cells.
    """
    h, w = grid.shape
    # Helpers to map between (r,c) and id
    to_id = lambda r, c: r * w + c
    to_rc = lambda idx: (idx // w, idx % w)

    # Count empty cells
    empties = np.argwhere(grid == 0)
    total_empty = len(empties)
    if number_of_obstacles >= total_empty:
        raise ValueError(f"Cannot add {number_of_obstacles} obstacles to {total_empty} empty cells")

    # Choose random starting empty cell
    sr, sc = empties[random.randrange(total_empty)]
    start_id = to_id(sr, sc)

    # Data structures for spanning tree
    parent = {start_id: None}
    children = {start_id: []}
    degree = {start_id: 0}
    visited = set([start_id])

    # Build randomized spanning tree via Prim's algorithm
    frontier = []  # list of (u_id, v_id)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Initialize frontier from start
    for dr, dc in directions:
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
            vid = to_id(nr, nc)
            frontier.append((start_id, vid))

    while frontier and len(visited) < total_empty:
        # Pick a random edge
        ui, vi = frontier.pop(random.randrange(len(frontier)))
        if vi in visited:
            continue
        # Add edge to tree
        parent[vi] = ui
        children[vi] = []
        children[ui].append(vi)
        # Update degrees
        degree[ui] = degree.get(ui, 0) + 1
        degree[vi] = degree.get(vi, 0) + 1
        visited.add(vi)
        # Add new edges from vi
        r, c = to_rc(vi)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                nid = to_id(nr, nc)
                if nid not in visited:
                    frontier.append((vi, nid))

    # Identify initial leaves (degree == 1)
    leaves = [nid for nid, deg in degree.items() if deg == 1]

    # Randomly prune leaves
    for _ in range(number_of_obstacles):
        if not leaves:
            break
        idx = random.randrange(len(leaves))
        node = leaves[idx]
        # Remove selected leaf from list
        last = leaves.pop()
        if node != last:
            leaves[idx] = last

        # Place obstacle
        r, c = to_rc(node)
        grid[r, c] = 1

        # Update neighbor degree and leaf list
        nbr = parent[node]
        if nbr is not None:
            degree[nbr] -= 1
            if degree[nbr] == 1:
                leaves.append(nbr)
        degree[node] = 0

    return grid

@numba.jit()
def get_grid_neighborhood(posX: int, posY: int, maze_grid: np.ndarray, food_sources: np.ndarray) -> np.ndarray:
    for y, x, _, food_present in food_sources:
        if food_present:
            maze_grid[y, x] = TileType.FOOD.value
        else:
            maze_grid[y, x] = TileType.FOOD_SOURCE.value
    neighborhood = np.zeros(10, dtype=np.int32)
    for i, dy, dx in [
        [0, -1, -1], [1, -1, 0], [2, -1, 1], [3, 0, -1], [4, 0, 1], [5, 1, -1], [6, 1, 0], [7, 1, 1]]:
        neighborhood[i] = maze_grid[posY + dy, posX + dx]
    for y, x, _, _ in food_sources:
        maze_grid[y, x] = TileType.EMPTY.value
    return neighborhood

class GridMazeWorld(gym.Env):
    def __init__(self, max_age=100, grid_size=12, obstacle_count=14, food_source_count=4,
                 food_generate_time_average=11, food_energy=12, initial_energy=20, max_energy=40,
                 energy_decay=0.98, energy_per_move=0.2, energy_per_time=0.8, name="something"):
        super(GridMazeWorld, self).__init__()

        self.max_age = max_age
        self.grid_size = grid_size
        self.obstacle_count = obstacle_count
        self.food_source_count = food_source_count
        self.food_energy = food_energy
        self.initial_energy = initial_energy
        self.max_energy = max_energy
        self.food_generate_time_average = food_generate_time_average
        self.name = name
        self.energy_decay = energy_decay
        self.energy_per_move = energy_per_move
        self.energy_per_time = energy_per_time

        self.tile_characters = [" ", "█", "X", "☺", "O"]
        self.tile_colors = [(32, 32, 32), (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 0, 255)]


        self.action_space = spaces.Discrete(len(Actions))
        # observation space should be 8 neighboring tiles of the agent, 1 for energy level
        self.observation_space = spaces.MultiDiscrete(
            [len(TileType)] * 8 + [len(TileType) + len(Actions), len(TileType) + len(Actions) + 6]
        )

        self.maze_grid = None
        self.food_source = None

        # initial position is a random empty tile
        self.posX, self.posY = None, None
        self.last_action = Actions.START.value

        self.energy = 0
        self.age = 0
        self.died = True

        self.reset()

    def max_observation_value(self):
        return len(TileType) + len(Actions) + 6

    def get_observation_size(self):
        return 10

    @staticmethod
    def generate_empy_maze_with_borders(size: int) -> np.ndarray:
        maze_grid = np.zeros((size, size), dtype=np.uint8)
        maze_grid[0, :] = TileType.OBSTACLE.value
        maze_grid[-1, :] = TileType.OBSTACLE.value
        maze_grid[:, 0] = TileType.OBSTACLE.value
        maze_grid[:, -1] = TileType.OBSTACLE.value
        return maze_grid

    @staticmethod
    def test_field(field):
        field = 1 - field
        p = np.nonzero(field)
        filled = flood_fill(field, (p[0][0], p[0][1]), 0, connectivity=1)
        return filled.sum() == 0

    def reset(self):
        self.maze_grid = self.generate_empy_maze_with_borders(self.grid_size)
        self.maze_grid = add_obstacles(self.maze_grid, self.obstacle_count)
        self.food_source = FoodSources(self.maze_grid, self.food_source_count, food_energy=self.food_energy,
                    food_generate_time_average=self.food_generate_time_average, food_generate_time_sdev=1)

        empty_tiles = np.argwhere(self.maze_grid == TileType.EMPTY.value)
        self.posX, self.posY = empty_tiles[np.random.choice(empty_tiles.shape[0])]


        self.died = False
        self.age = 0
        self.energy = self.initial_energy
        self.last_action = Actions.START.value

        return self.step(Actions.START.value)

    def step(self, action: int):
        if self.died:
            return self._create_observation(), 0, True, {}

        if action == Actions.LEFT.value and self.maze_grid[self.posY, self.posX - 1] == TileType.EMPTY.value:
            self.posX = self.posX - 1
            self.last_action = Actions.LEFT.value
        elif action == Actions.RIGHT.value and self.maze_grid[self.posY, self.posX + 1] == TileType.EMPTY.value:
            self.posX = self.posX + 1
            self.last_action = Actions.RIGHT.value
        elif action == Actions.UP.value and self.maze_grid[self.posY - 1, self.posX] == TileType.EMPTY.value:
            self.posY = self.posY - 1
            self.last_action = Actions.UP.value
        elif action == Actions.DOWN.value and self.maze_grid[self.posY + 1, self.posX] == TileType.EMPTY.value:
            self.posY = self.posY + 1
            self.last_action = Actions.DOWN.value
        else:
            self.last_action = Actions.STAY.value

        food_energy = food_step(self.posY, self.posX, self.food_source.food_sources, self.food_energy, self.food_generate_time_average, 1)
        self.energy = (self.energy_decay * self.energy) + food_energy - 1
        if self.last_action != Actions.STAY.value:
            self.energy -= self.energy_per_move
        self.energy = min(self.energy, self.max_energy)
        self.energy = max(self.energy, 0)
        self.age += 1
        self.died |= self.age >= self.max_age or self.energy <= 0

        step_reward = self.energy / self.max_energy

        maze_grid_observations = self._create_observation()

        return maze_grid_observations, step_reward, self.died , {}

    def _create_observation(self):
        maze_grid_observations = get_grid_neighborhood(self.posX, self.posY, self.maze_grid, self.food_source.food_sources)
        #maze_grid_observations = np.zeros(self.get_observation_size(), dtype=np.int32)
        maze_grid_observations[8] = len(TileType) + self.last_action
        maze_grid_observations[9] = len(TileType) + len(Actions) + int(self.energy * 5 / self.max_energy)
        return maze_grid_observations



    def add_obstacles(self, maze_grid: np.ndarray) -> np.ndarray:
        return add_obstacles_python(maze_grid, self.obstacle_count)

    def render(self, mode='text'):
        maze_grid = np.copy(self.maze_grid)
        maze_grid = self.food_source.render(maze_grid)
        maze_grid[self.posY, self.posX] = TileType.AGENT.value

        if mode == 'text':
            pic = np.full(maze_grid.shape, " ")
            for i in range(len(self.tile_characters)):
                pic[maze_grid == i] = self.tile_characters[i]

            print('\n\n\n\n')
            for line in pic:
                print(''.join([x for x in line]))
            print()
        else:
            patch_size = 32
            image = np.zeros([maze_grid.shape[0], maze_grid.shape[1], 3], dtype=np.uint8)
            for i in range(len(self.tile_colors)):
                image[maze_grid == i] = self.tile_colors[i]
            image = cv2.resize(image, (0, 0), fx=patch_size, fy=patch_size, interpolation=cv2.INTER_NEAREST)
            return image

    def close(self):
        ...


if __name__ == '__main__':
    max_age = 200
    obstacle_fraction = 0.33
    grid_size = 13
    obstacle_count = (grid_size - 2) * (grid_size - 2) * obstacle_fraction
    food_source_count = 8
    food_energy = 10
    initial_energy = 200

    env = GridMazeWorld(
        max_age=max_age,
        grid_size=grid_size,
        obstacle_count=int(obstacle_count + 0.5),
        food_source_count=food_source_count,
        food_energy=food_energy,
        initial_energy=initial_energy,
        max_energy=200,
        energy_decay=1,
        energy_per_move=0,
        energy_per_time=0,
    )
    cv2.namedWindow('env', cv2.WINDOW_NORMAL)
    env.reset()
    for i in tqdm(range(10000)):
        env.reset()
        for _ in range(max_age):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            #if done:
            #    break
            #print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Done: {done}")
            #cv2.imshow('env', env.render(mode=''))
            #cv2.waitKey(0)
    env.close()