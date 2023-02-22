from numba import jit
import numpy as np
import gym
import time
from gym import spaces
from skimage.morphology import flood_fill
import cv2

@jit
def add_food(food, field, food_count, size, food_energy):
    if (food > 0).sum() < food_count:
        pos = np.random.randint(0, size, size=2)
        if field[pos[0], pos[1]] == 0:
            food[pos[0], pos[1]] = food_energy
        return True
    return False


@jit
def try_move(field, posX, posY, dx, dy):
    x = min(field.shape[1] - 1, max(0, posX + dx))
    y = min(field.shape[0] - 1, max(0, posY + dy))

    if field[y, x] == 0:
        posX = x
        posY = y
    return posX, posY


@jit
def step(action, field, food, posX, posY, energy, age, food_count, food_energy, max_age, old_pos):
    if action == 0:
        x = max(0, posX - 1)
        if field[posY, x] == 0:
            posX = x
    elif action == 1:
        x = min(field.shape[1] - 1, posX + 1)
        if field[posY, x] == 0:
            posX = x
    elif action == 2:
        y = max(0, posY - 1)
        if field[y, posX] == 0:
            posY = y
    elif action == 3:
        y = min(field.shape[0] - 1, posY + 1)
        if field[y, posX] == 0:
            posY = y

    reward = food[posY, posX]
    energy += food[posY, posX] - 1
    food[posY, posX] = 0
    age += 1
    if (food > 0).sum() < food_count:
        pos = np.random.randint(0, field.shape[0], size=2)
        if field[pos[0], pos[1]] == 0:
            food[pos[0], pos[1]] = food_energy
    done = age >= max_age or energy <= 0
    if not done:
        reward += 1
    else:
        reward += energy

    observation = np.zeros((field.shape[0], field.shape[1], 4), dtype=np.uint8)
    observation[:, :, 0] = field * 255
    observation[:, :, 1] = (food > 0) * 255
    observation[posY, posX, 2] = 255
    observation[old_pos[1], old_pos[0], 3] = 125

    done = bool(done)
    reward /= 10
    return observation, reward, done, age, posX, posY, energy, age


class CustomEnv(gym.Env):
    def __init__(self, size=12, max_age=100, food_count=3, food_energy=12, obstacle_count=14):
        super(CustomEnv, self).__init__()

        self.size = size
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, shape=(size, size, 4), dtype=np.uint8)

        self.field = np.random.randint(0, 2, size=(size, size), dtype=np.uint8)
        self.food = np.zeros((size, size), dtype=np.uint8)

        self.posX = -1
        self.posY = -1
        self.old_pos = (0, 0)
        self.energy = 0
        self.age = 0

        self.max_age = max_age
        self.food_count = food_count
        self.food_energy = food_energy
        self.obstacle_count = obstacle_count
        self.reset()

    def step(self, action):
        observation, reward, done, self.age, self.posX, self.posY, self.energy, self.age = \
            step(action, self.field, self.food, self.posX, self.posY, self.energy, self.age, self.food_count,
                 self.food_energy, self.max_age, self.old_pos)
        self.old_pos = (self.posX, self.posY)
        return observation, reward, done, {}

    def _create_observation(self):
        observation = np.zeros((self.size, self.size, 4), dtype=np.uint8)
        observation[:, :, 0] = self.field * 255
        observation[:, :, 1] = (self.food > 0) * 255
        observation[self.posY, self.posX, 2] = 255
        observation[self.old_pos[1], self.old_pos[0], 3] = 255
        return observation

    def test_field(self, field):
        field = 1 - field
        p = np.nonzero(field)
        filled = flood_fill(field, (p[0][0], p[0][1]), 0, connectivity=1)
        return filled.sum() == 0

    def fill_blocks(self, field):
        to_fill = int(self.obstacle_count - field.sum())
        max_fill = to_fill
        while to_fill:
            new_field = np.copy(field)
            for pos in np.random.randint(0, field.shape[0], size=(to_fill, 2)):
                new_field[pos[0], pos[1]] = 1

            if self.test_field(new_field):
                field = new_field
            else:
                max_fill = max(1, to_fill // 2)
            to_fill = int(min(max_fill, self.obstacle_count - field.sum()))
        return field

    def reset(self):
        t1 = time.time()
        self.field[...] = 0
        self.food[...] = 0

        self.posX = -1
        self.posY = -1
        self.old_pos = (0, 0)

        self.field[...] = 0
        self.field = self.fill_blocks(self.field)
        while add_food(self.food, self.field, self.food_count, self.size, self.food_energy):
            pass
        while self.posX < 0:
            pos = np.random.randint(0, self.size, size=2)
            if self.field[pos[0], pos[1]] == 0:
                self.posX, self.posY = pos

        self.age = 0
        self.energy = self.food_energy

        return self._create_observation()

    def render(self, mode='text', field=None):
        if field is None:
            field = self.field
        if mode == 'text':
            observation = self._create_observation()
            pic = np.full((self.size, self.size), " ")
            pic[observation[:, :, 0] > 0] = "█"
            pic[observation[:, :, 1] > 0] = "X"
            pic[observation[:, :, 2] > 0] = "☺"

            print('\n\n\n\n')
            for line in pic:
                print(''.join([x for x in line]))
            print()
        else:
            patch_size = 32
            image = np.zeros((self.size*patch_size, self.size*patch_size, 3), dtype=np.uint8)
            image[:, :, 2] = cv2.resize(field, (image.shape[1], image.shape[1]), interpolation=cv2.INTER_NEAREST)
            image[:, :, 0] = cv2.resize(self.food, (image.shape[1], image.shape[1]), interpolation=cv2.INTER_NEAREST)
            image[self.posY*patch_size:(self.posY+1)*patch_size, self.posX*patch_size:(self.posX+1)*patch_size, 1] = 1
            image *= 255
            return image

    def close(self):
        ...