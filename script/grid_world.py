import time

from gym import Env, spaces
import numpy as np
import pygame

class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    CYAN = (0, 255, 255)


class GridWorld(Env):
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
    action2delta = {
        0: [1, 0],  # right
        1: [-1, 0], # left
        2: [0, 1],  # down
        3: [0, -1], # up
    }

    def __init__(self,
                 init_pos,
                 goal_pos,
                 human_pos,
                 grid,
                 action_success_rate=1.,
                 render=True,
                 cell_size_px=80,
                 cell_margin_px=5):
        # initialization values
        self.init_grid = np.copy(grid)
        self.init_pos = np.array(init_pos, dtype=int)
        self.goal_pos = np.array(goal_pos, dtype=int)
        self.human_pos = np.array(human_pos, dtype=int) if human_pos else None
        self.action_success_rate = action_success_rate

        # game state
        self.grid = np.copy(grid)
        self.agent_pos = np.array(init_pos, dtype=int)

        # pygame
        self.do_render = render
        self.cell_size = cell_size_px
        self.margin_size = cell_margin_px

        if self.do_render:
            pygame.init()
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
            self.render()

    def render(self, render_mode='human'):
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

            # human
            if self.human_pos is not None:
                color = GridWorld.feature2color['human']
                pygame.draw.circle(
                    self.surface,
                    color,
                    self._get_position_center(*self.human_pos),
                    int(self.cell_size / 2))

            # agent
            color = GridWorld.feature2color['agent']
            pygame.draw.circle(
                self.surface,
                color,
                self._get_position_center(*self.agent_pos),
                int(self.cell_size / 2))

            pygame.display.flip()

    def reset(self):
        np.copyto(dst=self.grid, src=self.init_grid)
        np.copyto(dst=self.agent_pos, src=self.init_pos)
        if self.do_render:
            self.render()

        obs = self.agent_pos[0] * self.agent_pos[1]
        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, info

    def step(self, action):
        if np.random.uniform() > self.action_success_rate:
            action = np.random.choice(4)
        delta = GridWorld.action2delta[action]
        if (np.all(self.agent_pos + delta >= [0, 0]) and
            np.all(self.agent_pos + delta < self.grid.shape)):
            self.agent_pos += delta
        self.render()

        obs = self.agent_pos[0] * self.agent_pos[1]
        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, info

    def action_space(self):
        return spaces.Discrete(4)

    def observation_space(self):
        return spaces.Discrete(self.grid.shape[0] * self.grid.shape[1])

    def _get_reward(self):
        # dense reward
        #dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        # sparse reward
        dist = 0 
        
        in_obs = GridWorld.enum2feature[
            self.grid[self.agent_pos[0], self.agent_pos[1]]] == 'obstacle'
        obs_penalty = in_obs * -10
        goal_reward = self._is_done() * 10

        reward = -dist + obs_penalty + goal_reward
        return reward

    def _is_done(self):
        return np.all(self.agent_pos == self.goal_pos)

    def _get_info(self):
        info_dict = dict(
            grid=self.grid,
            agent_pos=self.agent_pos,
            goal_pos=self.goal_pos,
            human_pos=self.human_pos)
        return info_dict

    def _get_position_topleft(self, i, j):
        x = (self.margin_size + self.cell_size) * i + self.margin_size
        y = (self.margin_size + self.cell_size) * j + self.margin_size
        return x, y

    def _get_position_center(self, i, j):
        x, y = self._get_position_topleft(i, j)
        x += int(self.cell_size / 2)
        y += int(self.cell_size / 2)
        return x, y


if __name__ == '__main__':
    grid = np.zeros((5, 5))
    grid[:4, 4] = 1
    env = GridWorld(
        init_pos=(0, 0),
        goal_pos=(4, 4),
        human_pos=(4, 0),
        render=True,
        grid=grid)
