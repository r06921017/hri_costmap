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
    ''' Grid world with pygame rendering.'''
    feature2color = {
        'empty': Colors.WHITE,
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
                 dimensions,
                 init_pos,
                 goal_pos,
                 reward_grid,
                 human_pos=None,
                 action_success_rate=1.,
                 render=True,
                 cell_size_px=80,
                 cell_margin_px=5):
        '''
        :param dimensions: dimensions of gridworld as (N, M) tuple
        :param init_pos: starting position as a vector OR a probability
            distribution over the state space as a 2D array.
        :param goal_pos: goal positions as a list of vectors.
        :param reward_grid: reward function over states, as a 2D array.
        :param human_pos: human position as a list of vectors (optional).
        :param action_success_rate: probability a action will move the agent to
            the expected next state. Set to 1.0 for deterministic transitions.
        :param render: boolean flag to render the pygame window.
        :param cell_size_px: side length of a grid cell in pygame.
        :param cell_margin_px: margin between grid cells in pygame.
        '''
        # initialization values
        self.N, self.M = dimensions
        self.init_pos = np.asarray(init_pos)
        self.goal_pos = np.asarray(goal_pos).reshape((-1, 2))
        self.reward = np.asarray(reward_grid).flatten()
        if human_pos is not None:
            self.human_pos = np.asarray(human_pos).reshape((-1, 2))
        else:
            self.human_pos = None
        self.action_success_rate = action_success_rate

        # sanity check
        assert self.N > 0 and self.M > 0
        if self.init_pos.shape == (2,):
            # single start position
            assert self._valid_pos(self.init_pos)
        else:
            # probability distribution
            assert (self.init_pos.shape == (N, M) and
                    np.sum(init_pos) == 1 and
                    np.all(init_pos) >= 0)
        assert np.all(self._valid_pos_vec(self.goal_pos))
        assert self.reward.shape == (self.N * self.M,)
        if self.human_pos is not None:
            assert np.all(self._valid_pos_vec(self.human_pos))
        assert self.action_success_rate >= 0 and self.action_success_rate <= 1

        # game state
        self.agent_state = self._pos2state(self.init_pos)
        self.goal_states = self._pos2state_vec(self.goal_pos)
        self.transition = self._construct_transition_fn()

        # pygame
        self.do_render = render
        self.cell_size = cell_size_px
        self.margin_size = cell_margin_px

        if self.do_render:
            pygame.init()
            screen_width_px = ((self.cell_size + self.margin_size) * self.N +
                self.margin_size)
            screen_height_px = ((self.cell_size + self.margin_size) * self.M +
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
            for i in range(self.N):
                for j in range(self.M):
                    rect = pygame.Rect(
                        self._get_position_topleft(i, j)[0],
                        self._get_position_topleft(i, j)[1],
                        self.cell_size,
                        self.cell_size)
                    color = GridWorld.feature2color['empty']
                    pygame.draw.rect(self.surface, color, rect)

            # goals
            color = GridWorld.feature2color['goal']
            for g in self.goal_pos:
                rect = pygame.Rect(
                    self._get_position_topleft(*g)[0],
                    self._get_position_topleft(*g)[1],
                    self.cell_size,
                    self.cell_size)
                pygame.draw.rect(self.surface, color, rect)

            # humans
            if self.human_pos is not None:
                for h in self.human_pos:
                    color = GridWorld.feature2color['human']
                    pygame.draw.circle(
                        self.surface,
                        color,
                        self._get_position_center(*h),
                        int(self.cell_size / 2))

            # agent
            color = GridWorld.feature2color['agent']
            agent_pos = self._get_position_center(
                *self._state2pos(self.agent_state))
            pygame.draw.circle(
                self.surface,
                color,
                agent_pos,
                int(self.cell_size / 2))

            pygame.display.flip()

    def reset(self):
        if self.init_pos.shape == (self.N, self.M):
            self.agent_state = np.random.choice(
                self.N * self.M, p=self.init_pos.flatten())
        else:
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
        sprime_prob = self.transition[action][self.agent_state]
        # decompress sparse array
        if hasattr(sprime_prob, 'toarray'):
            sprime_prob = sprime_prob.toarray().flatten()
        self.agent_state = np.random.choice(len(sprime_prob), p=sprime_prob)
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
        return gym.spaces.Discrete(self.N * self.M)

    def _get_reward(self):
        return self.reward[self.agent_state]

    def _is_done(self):
        return self.agent_state in self.goal_states

    def _get_info(self):
        info_dict = dict(
            agent_pos=self._state2pos(self.agent_state),
            goal_pos=self.goal_pos,
            human_pos=self.human_pos,
            reward_grid=self.reward.reshape((self.N, self.M)))
        return info_dict

    def _construct_transition_fn(self):
        # T[a][s, s'] implemented as a length |A| list of sparse SxS matrices
        S, A = self.observation_space().n, self.action_space().n
        T = [csr_matrix((S, S), dtype=np.float) for _ in range(A)]
        
        for i in range(self.N):
            for j in range(self.M):
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

        # block transitions out of goal states
        for goal_state in self.goal_states:
            sprime_prob = np.zeros((S,))
            sprime_prob[goal_state] = 1
            for a in range(self.action_space().n):
                T[a][goal_state, :] = sprime_prob

        return T

    def _get_position_topleft(self, i, j):
        x = (self.margin_size + self.cell_size) * i + self.margin_size
        y = (self.margin_size + self.cell_size) * j + self.margin_size
        return x, y

    def _get_position_center(self, i, j):
        x, y = self._get_position_topleft(i, j)
        x += int(self.cell_size / 2)
        y += int(self.cell_size / 2)
        return x, y

    def _state2pos(self, s):
        return np.unravel_index(s, (self.N, self.M))

    def _pos2state(self, coords):
        return np.ravel_multi_index(coords, (self.N, self.M))

    def _valid_pos(self, coords):
        return (coords[0] >= 0 and coords[0] < self.N and
                coords[1] >= 0 and coords[1] < self.M)

    def _state2pos_vec(self, sa):
        return np.array([self._state2pos(s) for s in sa])

    def _pos2state_vec(self, ca):
        return np.array([self._pos2state(c) for c in ca])

    def _valid_pos_vec(self, coord_list):
        return [self._valid_pos(coord) for coord in coord_list]

    def _feature_map(self, state):
        # TODO: move out of gridworld
        feat = np.zeros(self.observation_space().n)
        feat[state] = 1
        return feat
