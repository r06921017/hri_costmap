from itertools import product

import numpy as np


def construct_goal_reward(grid, goal_pos, goal_reward):
    '''
    Applies a reward at the given goal positions.

    :param grid: NxM ndarray for the reward grid.
    :param goal_pos: goal positions as a Gx2 ndarray.
    :param goal_rew: a scalar for the goal reward OR a list of scalars that
        match the first dimension of goal_pos.
    '''
    goals = goal_pos.reshape((-1, 2))
    rewards = broadcast(goals, goal_reward)

    for goal, rew in zip(goals, rewards):
        grid[goal[0], goal[1]] += rew

    return grid

def construct_human_radius_reward(grid, human_pos, human_radius, human_reward):
    '''
    Applies a flat reward at a fixed radius around a list of humans.

    :param grid: NxM array for the reward grid.
    :param human_pos: human positions as a Hx2 ndarray.
    :param human radius: a scalar for the radius around the human OR a list of
        scalars that match the first dimension of human_pos.
    :param human radius: a scalar for the reward within a human's radius OR a
        list of scalars that match the first dimension of human_pos
    '''
    humans = human_pos.reshape((-1, 2))
    radii = broadcast(humans, human_radius)
    rewards = broadcast(humans, human_reward)

    for human, radius, reward in zip(human_pos, radii, rewards):
        for i, j in product(range(grid.shape[0]), range(grid.shape[1])):
            p = np.array([i, j])
            if np.linalg.norm(p - human) < radius:
                grid[i, j] += reward

    return grid

def construct_feature_boundary_reward(
    grid, boundary_axis, boundary_value, reward, exp_constant):
    '''
    Applies a feature-based costmap reward structure to the grid.

    :param grid: NxM array for the reward grid.
    :param boundary_axis: axis along which to apply the feature reward.
    :param boundary_value: reward will be applied along boundary_axis up
        until this value.
    :param reward: Base reward at boundary_axis=0. Reward will be
        exponentially decayed as the non-boundary_axis value increases.
    :param exp_constant: exponential decay constant.
    '''
    for i, j in product(range(grid.shape[0]), range(grid.shape[1])):
        if boundary_axis == 0 and i < boundary_value:
            grid[i, j] += reward * np.exp(-exp_constant * j)
        elif boundary_axis == 1 and j < boundary_value:
            grid[i, j] += reward * np.exp(-exp_constant * i)

    return grid

def broadcast(primary, secondary):
    '''
    Broadcast a secondary scalar to the primary array's first dimension.
    '''
    if hasattr(secondary, 'len'):
        return secondary
    else:
        secondary_broadcasted = np.array([secondary] * primary.shape[0])
        return secondary_broadcasted
