from numba import jit
import numpy as np
import gym
import time
from gym import spaces
from skimage.morphology import flood_fill
import cv2
import random
from tqdm import tqdm


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
        self.food_sources: list[tuple[int, int, int, bool]] = []
        self.add_food_sources(maze_grid)

    def add_food_sources(self, maze_grid: np.ndarray):
        empty_tile_coordinates = np.argwhere(maze_grid == TileType.EMPTY.value)
        empty_tile_coordinates = empty_tile_coordinates[np.random.choice(empty_tile_coordinates.shape[0], self.food_source_count, replace=False)]
        for y, x in empty_tile_coordinates:
            self.food_sources.append((y, x, 0, False))

    def step(self, agent_y: int, agen_x: int):
        # if the agent is on a food source with food, eat it and generate the number of time steps to generate new food
        # Check if some food should be generated
        food_energy = 0
        for i, (y, x, time_to_generate_food, food_present) in enumerate(self.food_sources):
            if agent_y == y and agen_x == x:
                if food_present:
                    food_energy += self.food_energy
                    time_to_generate_food = 1 + int(0.5 + random.gauss(self.food_generate_time_average, self.food_generate_time_sdev))
                    self.food_sources[i] = (y, x, time_to_generate_food, False)
            elif time_to_generate_food == 0:
                self.food_sources[i] = (y, x, -1, True) # generate food
            elif time_to_generate_food > 0:
                self.food_sources[i] = (y, x, time_to_generate_food-1, False)
        return food_energy

    def render(self, maze_grid: np.ndarray):
        for y, x, time_to_generate_food, food_present in self.food_sources:
            if food_present:
                maze_grid[y, x] = TileType.FOOD.value
            else:
                maze_grid[y, x] = TileType.FOOD_SOURCE.value
        return maze_grid



class GridMazeWorld(gym.Env):
    def __init__(self, max_age=100, grid_size=12, obstacle_count=14, food_source_count=4,
                 food_generate_time_average=11, food_energy=12, initial_energy=20, max_energy=40, name="something"):
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

        self.neighborhood = np.array(
            [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int8)

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
        self.maze_grid = self.add_obstacles(self.maze_grid)
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

        food_energy = self.food_source.step(self.posY, self.posX)
        self.energy += food_energy - 1
        self.energy = min(self.energy, self.max_energy)
        self.energy = max(self.energy, 0)
        self.age += 1
        self.died |= self.age >= self.max_age or self.energy <= 0

        step_reward = self.energy

        maze_grid_observations = self._create_observation()

        step_reward /= self.max_energy
        return maze_grid_observations, step_reward, self.died , {}

    def _create_observation(self):
        maze_grid = np.copy(self.maze_grid)
        self.food_source.render(maze_grid)

        maze_grid_observations = [maze_grid[self.posY + dy, self.posX + dx] for dy, dx in self.neighborhood]
        maze_grid_observations.append(len(TileType) + self.last_action)
        maze_grid_observations.append(len(TileType) + len(Actions) + int(self.energy * 5 / self.max_energy))
        return maze_grid_observations

    def add_obstacles(self, maze_grid: np.ndarray) -> np.ndarray:
        to_fill = self.obstacle_count
        empty_tiles = np.argwhere(self.maze_grid == TileType.EMPTY.value)
        empty_tiles = np.random.permutation(empty_tiles)
        while empty_tiles.size != 0 and to_fill > 0:
            next_tile = empty_tiles[0]
            empty_tiles = empty_tiles[1:]
            new_maze_grid = np.copy(maze_grid)
            new_maze_grid[next_tile[0], next_tile[1]] = TileType.OBSTACLE.value

            if self.test_field(new_maze_grid):
                maze_grid = new_maze_grid
                to_fill -= 1

        return maze_grid

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
    max_age = 100
    obstacle_fraction = 0.33
    grid_size = 11
    obstacle_count = (grid_size - 2) * (grid_size - 2) * obstacle_fraction
    food_source_count = 10
    food_energy = 10
    initial_energy = 20

    env = GridMazeWorld(
        max_age=max_age,
        grid_size=grid_size,
        obstacle_count=obstacle_count,
        food_source_count=food_source_count,
        food_energy=food_energy,
        initial_energy=initial_energy,
    )
    cv2.namedWindow('env', cv2.WINDOW_NORMAL)
    for i in tqdm(range(1000)):
        env.reset()
        for _ in range(max_age):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
            #print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Done: {done}")
            #cv2.imshow('env', env.render(mode=''))
            #cv2.waitKey(0)

            #
    env.close()