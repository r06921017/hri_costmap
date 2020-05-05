#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import gym
import time
import typing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.sparse import csr_matrix

from max_ent_irl import MaxEntIRL
from util.plotting import plot_grid_map, plot_policy
from util.reward import construct_goal_reward, construct_human_radius_reward
from util.dataset import Dataset, transition


class CostmapIRL:
    def __init__(self):
        self.csv_path = '/home/rdaneel/hri_costmap_traj/actions/'
        self.df_list = list()
        self.file_name = list()
        self.goal_states = list()

        for _fin in os.listdir(self.csv_path):
            self.df_list.append(pd.read_csv(self.csv_path + _fin))
            self.file_name.append(_fin)

        # Get max length
        self.max_length = 0
        for _df in self.df_list:
            if _df.shape[0] > self.max_length:
                self.max_length = _df.shape[0]
        
        # Get the maximum position among all demos
        tmp_f1 = list()
        tmp_f2 = list()
        tmp_a1 = list()
        tmp_a2 = list()
        for i, _df in enumerate(self.df_list):
            tmp_f1.append(np.nanmax(_df['f1'].tolist()))
            tmp_f2.append(np.nanmax(_df['f2'].tolist()))
            tmp_a1.append(np.nanmax(_df['a1'].tolist()))
            tmp_a2.append(np.nanmax(_df['a2'].tolist()))

        self.max_f1 = np.around(np.nanmax(tmp_f1), 1)
        self.max_f2 = np.around(np.nanmax(tmp_f2), 1)
        self.max_a1 = np.around(np.nanmax(tmp_a1), 1)
        self.max_a2 = np.around(np.nanmax(tmp_a2), 1)

        # Get the minimum position among all demos
        tmp_f1 = list()
        tmp_f2 = list()
        tmp_a1 = list()
        tmp_a2 = list()
        for _df in self.df_list:
            tmp_f1.append(np.nanmin(_df['f1'].tolist()))
            tmp_f2.append(np.nanmin(_df['f2'].tolist()))
            tmp_a1.append(np.nanmin(_df['a1'].tolist()))
            tmp_a2.append(np.nanmin(_df['a2'].tolist()))

        self.min_f1 = np.around(np.nanmin(tmp_f1),1)
        self.min_f2 = np.around(np.nanmin(tmp_f2),1)
        self.min_a1 = np.around(np.nanmin(tmp_a1),1)
        self.min_a2 = np.around(np.nanmin(tmp_a2),1)

        self.f1_idx = dict()
        for i, f in enumerate(np.arange(self.min_f1, self.max_f1 + 0.05, 0.1)):
            self.f1_idx[i] = np.around(f, 1)
        self.f1_length = len(self.f1_idx)
        self.inv_f1_idx = {_tmp_v: _tmp_k for _tmp_k, _tmp_v in self.f1_idx.items()}

        self.f2_idx = dict()
        for i, f in enumerate(np.arange(self.min_f2, self.max_f2 + 0.05, 0.1)):
            self.f2_idx[i] = np.around(f, 1)
        self.f2_length = len(self.f2_idx)
        self.inv_f2_idx = {_tmp_v: _tmp_k for _tmp_k, _tmp_v in self.f2_idx.items()}

        print('max: ({0}, {1}, {2}, {3})'.format(self.max_f1, self.max_f2, self.max_a1, self.max_a2))
        print('min: ({0}, {1}, {2}, {3})'.format(self.min_f1, self.min_f2, self.min_a1, self.min_a2))

        # Get goal positions
        self.goal_states = np.array([self._pos2state(np.around(_f1, 1), 0.0) 
            for _f1 in np.arange(self.min_f1, self.max_f1+0.05, 0.1)])

        # viewing manhattan distance of robot to goal and robot to human as states
        self.state_size = self.f1_length * self.f2_length
        print('state space size: {0}'.format(self.state_size))

        self.action_idx = dict()
        for i, a1 in enumerate(np.arange(self.min_a1, self.max_a1 + 0.05, 0.1)):
            for j, a2 in enumerate(np.arange(self.min_a2, self.max_a2 + 0.1, 0.1)):
                a_id = i*len(np.arange(self.min_a2, self.max_a2 + 0.05, 0.1)) + j
                self.action_idx[a_id] = (np.around(a1, 1), np.around(a2, 1))

        self.inv_action_idx = {_tmp_v: _tmp_k for _tmp_k, _tmp_v in self.action_idx.items()}

        S, A = self.observation_space().n, self.action_space().n
        self.T = [csr_matrix((S, S), dtype=np.float) for _ in range(A)]
        
    
    def _construct_transition_fn(self):
        # T[a][s, s'] implemented as a length |A| list of sparse SxS matrices
        S, A = self.observation_space().n, self.action_space().n
        T = [csr_matrix((S, S), dtype=np.float) for _ in range(A)]
        
        # for i in range(self.f1_length):
        #     for j in range(self.f2_length):
        #         for a, delta in enumerate(GridWorld.action2delta):
        #             # state data
        #             s_pos = np.array([i, j])
        #             s = self._pos2state(s_pos)

        #             # get valid neighbors (current state is also valid)
        #             neighbors = [s_pos + d for d in
        #                 GridWorld.action2delta + [[0, 0]]]
        #             valid_neighbors = [self._pos2state(n) for n in neighbors
        #                 if self._valid_pos(n)]

        #             # if sprime_pos is not valid, clip to current state
        #             sprime_pos = s_pos + delta
        #             if self._valid_pos(sprime_pos):
        #                 success_state = self._pos2state(sprime_pos)
        #             else:
        #                 success_state = s

        #             # uniform weighting of all other valid transitions
        #             fail_prob = (1 - self.action_success_rate)
        #             if len(valid_neighbors) > 1:
        #                 fail_prob /= len(valid_neighbors) - 1

        #             for n in valid_neighbors:
        #                 if n == success_state:
        #                     T[a][s, success_state] = self.action_success_rate
        #                 else:
        #                     T[a][s, n] = fail_prob

        # # block transitions out of goal states
        # for goal_state in self.goal_states:
        #     sprime_prob = np.zeros((S,))
        #     sprime_prob[goal_state] = 1
        #     for a in range(self.action_space().n):
        #         T[a][goal_state, :] = sprime_prob

        return T


    def _pos2state(self, f1, f2):
        return int(self.inv_f1_idx[f1]*self.f2_length + self.inv_f2_idx[f2])

    def _state2pos(self, idx):
        f1 = self.f1_idx[idx // self.f2_length]
        f2 = self.f2_idx[idx % self.f2_length]

        return f1, f2;

    def _transition(self, obs, in_action, next_obs):
        cur_f1, cur_f2 = self._state2pos(obs)
        act1 = self.action_idx[in_action][0]
        act2 = self.action_idx[in_action][1]

        tmp_f1 = cur_f1 + act1
        tmp_f2 = cur_f2 + act2

        next_f1, next_f2 = self._state2pos(next_obs)
        
        # Deterministic, probability = 1
        if tmp_f1 == next_f1 and tmp_f2 == next_f2:
            return 1.
        else:
            return 0.

    def observation_space(self):
        return gym.spaces.Discrete(self.state_size)

    def action_space(self):
        return gym.spaces.Discrete(len(self.action_idx))

    def _feature_map(self, state):
        # TODO: move out of gridworld
        feat = np.zeros(self.observation_space().n)
        feat[state] = 1
        return feat

    
    def collect_demo(self):        
        dataset = Dataset(self.max_length)
        for _df in self.df_list:
            t = list()
            for idx in range(_df.shape[0]-1):  # before termination?
                t.append(transition(
                    obs=self._pos2state(np.around(_df['f1'][idx], 1), np.around(_df['f2'][idx], 1)), 
                    act=self.inv_action_idx[(np.around(_df['a1'][idx+1], 1), np.around(_df['a2'][idx+1], 1))],
                    next_obs=self._pos2state(np.around(_df['f1'][idx+1], 1), np.around(_df['f2'][idx+1], 1)),
                    rew=1.0))
            dataset.append(t)
        return dataset
                

    def main(self):
        # phi
        phi = [self._feature_map(s) for s in range(self.observation_space().n)]
        phi = np.array(phi)

        # IRL
        me_irl = MaxEntIRL(
            observation_space=self.observation_space(),
            action_space=self.action_space(),
            transition=self._transition,
            goal_states=self.goal_states,
            dataset=self.collect_demo(),
            feature_map=phi,
            max_iter=10,
            anneal_rate=0.9)
        Rprime = me_irl.train()
        Rprime = Rprime.reshape((self.f1_length, self.f2_length)).T

        # plot results
        plot_grid_map(Rprime, "Reward (IRL)", print_values=True, cmap=plt.cm.Blues)
        plt.show()
        return
    
if __name__ == '__main__':
    costmap_irl = CostmapIRL()
    costmap_irl.main()