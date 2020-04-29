import time

import gym
import numpy as np
import pygame
from scipy.sparse import csr_matrix


class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    CYAN = (0, 255, 255)


class GridWorld(gym.Env):
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
    action2delta = [
        [1, 0],  # right
        [-1, 0], # left
        [0, 1],  # down
        [0, -1], # up
    ]
    metadata = {'render.modes': ['human',]}  # for gym.Env

    def __init__(self,
                 init_pos,
                 goal_pos,
                 grid,
                 human_pos=None,
                 human_radius=2.5,
                 action_success_rate=1.,
                 render=True,
                 cell_size_px=80,
                 cell_margin_px=5):
        # initialization values
        self.init_grid = np.copy(grid)
        self.init_pos = np.array(init_pos, dtype=int)
        self.goal_pos = np.array(goal_pos, dtype=int)
        self.human_pos = np.array(human_pos, dtype=int) if human_pos else None
        self.human_radius = human_radius
        self.action_success_rate = action_success_rate

        # game state
        self.grid = np.copy(self.init_grid)
        self.agent_state = self._pos2state(self.init_pos)
        self.goal_state = self._pos2state(self.goal_pos)
        self.R = self._construct_reward_fn()
        self.T = self._construct_transition_fn()

        # sanity check
        assert self._valid_pos(self.init_pos)
        assert self._valid_pos(self.goal_pos)
        assert (self._valid_pos(self.human_pos) if
                self.human_pos is not None else True)
        assert self.human_radius > 0.
        assert (self.action_success_rate >= 0 and
                self.action_success_rate <= 1)

        # pygame
        self.do_render = render
        self.cell_size = cell_size_px
        self.margin_size = cell_margin_px

        if self.do_render:
            pygame.init()
            screen_width_px = (
                (self.cell_size + self.margin_size) * self.grid.shape[0] +
                self.margin_size)
            screen_height_px = (
                (self.cell_size + self.margin_size) * self.grid.shape[1] +
                self.margin_size)
            self.surface = pygame.display.set_mode(
                (screen_width_px, screen_height_px),
                # flags=pygame.RESIZABLE  # TODO: fix rendering when resized
            )
            self.render()

    def render(self, mode='human'):
        if mode != 'human':
            super(GridWorld, self).render(mode=mode)

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
                    feature = GridWorld.enum2feature[self.grid[i, j]]
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
                self._get_position_center(*self._state2pos(self.agent_state)),
                int(self.cell_size / 2))

            pygame.display.flip()

    def reset(self):
        np.copyto(dst=self.grid, src=self.init_grid)
        self.agent_state = self._pos2state(self.init_pos)
        if self.do_render:
            self.render()

        obs = self.agent_state
        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, info

    def close(self):
        if pygame.display.get_init():
            pygame.display.quit()
        pygame.quit()

    def step(self, action):
        sprime_prob = self.T[action][self.agent_state].toarray().flatten()
        self.agent_state = np.random.choice(self.grid.size, p=sprime_prob)
        if self.do_render:
            self.render()

        obs = self.agent_state
        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, info

    def action_space(self):
        return gym.spaces.Discrete(len(GridWorld.action2delta))

    def observation_space(self):
        return gym.spaces.Discrete(self.grid.size)

    def _get_reward(self):
        return self.R[self.agent_state]

    def _is_done(self):
        return self.agent_state == self.goal_state

    def _get_info(self):
        info_dict = dict(
            grid=self.grid,
            agent_pos=self._state2pos(self.agent_state),
            goal_pos=self.goal_pos,
            human_pos=self.human_pos)
        return info_dict

    def _state2pos(self, s):
        return np.unravel_index(s, self.grid.shape)

    def _pos2state(self, coords):
        return np.ravel_multi_index(coords, self.grid.shape)

    def _get_position_topleft(self, i, j):
        x = (self.margin_size + self.cell_size) * i + self.margin_size
        y = (self.margin_size + self.cell_size) * j + self.margin_size
        return x, y

    def _get_position_center(self, i, j):
        x, y = self._get_position_topleft(i, j)
        x += int(self.cell_size / 2)
        y += int(self.cell_size / 2)
        return x, y

    def _construct_reward_fn(self):
        # R[s]
        R = np.zeros(self.grid.size)

        for s, v in zip(range(self.grid.size), self.grid.flat):
            # nongoal penalty
            if GridWorld.enum2feature[v] == 'empty':
                R[s] += -1
            # obstacle penalty
            if GridWorld.enum2feature[v] == 'obstacle':
                R[s] += -10
            # human proximity penalty
            s_pos = self._state2pos(s)
            if (self.human_pos is not None and
                np.linalg.norm(s_pos - self.human_pos) < self.human_radius):
                R[s] += -10
        # goal reward
        R[self.goal_state] = 10

        return R

    def _construct_transition_fn(self):
        # T[a][s, s'] implemented as a length |A| list of sparse SxS matrices
        T = [csr_matrix((self.grid.size, self.grid.size), dtype=np.float)
             for _ in range(self.action_space().n)]
        
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                for a, delta in enumerate(GridWorld.action2delta):
                    # state data
                    s_pos = np.array([i, j])
                    s = self._pos2state(s_pos)

                    # get valid neighbors (current state is also valid)
                    neighbors = [s_pos + d for d in
                        GridWorld.action2delta + [[0, 0]]]
                    valid_neighbors = [self._pos2state(n) for n in neighbors
                        if self._valid_pos(n)]

                    # if sprime_pos is not valid, clip to current state
                    sprime_pos = s_pos + delta
                    if self._valid_pos(sprime_pos):
                        success_state = self._pos2state(sprime_pos)
                    else:
                        success_state = s

                    # uniform weighting of all other valid transitions
                    fail_prob = (1 - self.action_success_rate)
                    if len(valid_neighbors) > 1:
                        fail_prob /= len(valid_neighbors) - 1

                    for n in valid_neighbors:
                        if n == success_state:
                            T[a][s, success_state] = self.action_success_rate
                        else:
                            T[a][s, n] = fail_prob

        return T

    def _valid_pos(self, coords):
        return (coords[0] >= 0 and coords[0] < self.grid.shape[0] and
                coords[1] >= 0 and coords[1] < self.grid.shape[1])

    def _feature_map(self, state):
        max_dist = np.linalg.norm(self.init_pos - self.goal_pos)
        dist_goal = -np.linalg.norm(self.goal_pos - self._state2pos(state)) / max_dist
        dist_human = np.linalg.norm(self.human_pos - self._state2pos(state)) / max_dist

        return [dist_goal, dist_human]
