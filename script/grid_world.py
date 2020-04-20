import numpy as np
import pygame
import time


class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    CYAN = (0, 255, 255)


class GridWorld:
    ''' Pygame-based grid world.'''
    enum2feature = {
        0: 'empty',
        1: 'obstacle',
        2: 'goal',
        3: 'human',
    }
    feature2color = {
        'empty': Colors.WHITE,
        'obstacle': Colors.RED,
        'goal': Colors.GREEN,
        'agent': Colors.BLUE,
        'human': Colors.YELLOW,
    }

    def __init__(self,
                 init_pos,
                 goal_pos,
                 human_pos,
                 grid,
                 cell_size_px=80,
                 cell_margin_px=5):
        # initialization values
        self.init_grid = np.copy(grid)
        self.init_pos = np.asarray(init_pos)
        self.goal_pos = np.asarray(goal_pos)
        self.human_pos = np.asarray(human_pos)

        # game state
        self.grid = np.copy(grid)
        self.agent_pos = np.asarray(init_pos)

        # pygame
        self.cell_size = cell_size_px
        self.margin_size = cell_margin_px

        pygame.init()
        pygame.font.init()

        screen_width_px = (
            (self.cell_size + self.margin_size) * grid.shape[0] +
            self.margin_size)
        screen_height_px = (
            (self.cell_size + self.margin_size) * grid.shape[1] +
            self.margin_size)
        self.surface = pygame.display.set_mode(
            (screen_width_px, screen_height_px),
            # flags=pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.render()

    def render(self):
        if pygame.display.get_init():
            self.surface.fill(Colors.BLACK)
            # grid
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    rect = pygame.Rect(
                        self._get_position_topleft(i, j)[0],
                        self._get_position_topleft(i, j)[1],
                        self.cell_size,
                        self.cell_size)
                    feature = GridWorld.enum2feature[grid[i, j]]
                    color = GridWorld.feature2color[feature]
                    pygame.draw.rect(self.surface, color, rect)

            # goal
            color = GridWorld.feature2color['goal']
            rect = pygame.Rect(
                self._get_position_topleft(*self.goal_pos)[0],
                self._get_position_topleft(*self.goal_pos)[1],
                self.cell_size,
                self.cell_size)
            pygame.draw.rect(self.surface, color, rect)

            # agent
            color = GridWorld.feature2color['agent']
            pygame.draw.circle(
                self.surface,
                color,
                self._get_position_center(*self.agent_pos),
                int(self.cell_size / 2))

            # human
            color = GridWorld.feature2color['human']
            pygame.draw.circle(
                self.surface,
                color,
                self._get_position_center(*self.human_pos),
                int(self.cell_size / 2))

            pygame.display.flip()

    def reset(self):
        np.copyto(dst=self.grid, src=self.init_grid)
        self.agent_pos = self.init_pos
        self.render()

        obs = self.agent_pos
        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, info


    def step(self, action):
        # TODO sanitize
        self.agent_pos += action
        self.render()

        obs = self.agent_pos
        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, info

    def _get_reward(self):
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        if GridWorld.enum2feature[
                self.grid[self.agent_pos[0], self.agent_pos[1]]] == 'obstacle':
            obs_penalty = -10
        else:
            obs_penalty = 0

        if self._is_done():
            goal_reward = 10
        else:
            goal_reward = 0

        reward = -dist + obs_penalty + goal_reward
        return reward

    def _is_done(self):
        return np.all(self.agent_pos == self.goal_pos)

    def _get_info(self):
        return {}

    def _get_position_topleft(self, i, j):
        x = (self.margin_size + self.cell_size) * i + self.margin_size
        y = (self.margin_size + self.cell_size) * j + self.margin_size
        return x, y

    def _get_position_center(self, i, j):
        x, y = self._get_position_topleft(i, j)
        x += int(self.cell_size / 2)
        y += int(self.cell_size / 2)
        return x, y


grid = np.zeros((5, 5))
grid[:4, 4] = 1
world = GridWorld(
    init_pos=(0, 0),
    goal_pos=(4, 4),
    human_pos=(4, 0),
    grid=grid)
