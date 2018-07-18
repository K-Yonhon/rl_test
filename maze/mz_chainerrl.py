# coding=utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np

# https://qiita.com/masataka46/items/7729a74d8dc15de7b5a8
# https://qiita.com/ys-0-sy/items/404cc5388e9d77e16734

env = gym.make('CartPole-v0')
# print('observation space:', env.observation_space)
# print('action space:', env.action_space)
#
# obs = env.reset()
# # env.render()
# print('initial observation:', obs)
#
# action = env.action_space.sample()
# obs, r, done, info = env.step(action)
# print('next observation:', obs)
# print('reward:', r)
# print('done:', done)
# print('info:', info)


class QFunction(chainer.Chain):
    # def __init__(self, obs_size, n_actions, n_hidden_channels=50):
    #     super().__init__(
    #         l0=L.Linear(obs_size, n_hidden_channels),
    #         l1=L.Linear(n_hidden_channels, n_hidden_channels),
    #         l2=L.Linear(n_hidden_channels, n_actions))
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)
# q_func.to_qpu()