# coding=utf-8

import gym
import gym.spaces
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from keras.models import load_model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from keras.layers.core import Dropout
from rl.agents import DDPGAgent
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy
from rl.core import Processor
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, MaxPooling2D
# from operator import truediv

from mz_data import MzData


class Mz(gym.core.Env):
    def __init__(self):
        self.AGENT = 0.5
        self.FLAG = 0.65
        self.PATH = 1

        mz_data = MzData()
        mz_data.read(csv='data30x30.csv')
        self.flags = mz_data.flags
        self.c_size = mz_data.shape[1]
        self.r_size = mz_data.shape[0]
        self.start = mz_data.start
        self.goal = mz_data.goal

        self.maze = np.ones(mz_data.shape)

        # self.pos = [0, 0]
        self.st = None
        self.actions = {
            0: (0, 1),
            1: (0, -1),
            2: (1, 0),
            3: (-1, 0)
        }

        # rigtht, left, up, down,
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.max_index = self.c_size-1
        self.a_self = self.c_size

        self.check_flags = self.flags.copy()

        # self.flg_data = np.ones((len(self.flags), ))
        # flg_pos_list = [[row, col] for row, col in self.flags]
        # self.flg_pos = np.array(flg_pos_list)
        self.flg_dict = {}
        for i, flg_pos in enumerate(self.flags):
            self.flg_dict[tuple(flg_pos)] = i

        # low = np.empty(self.maze.shape)
        # high = np.empty(self.maze.shape)
        # low.fill(self.AGENT)
        # high.fill(self.PATH)
        low = self.AGENT
        high = self.PATH
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=self.maze.shape, dtype=np.float32)

    def step(self, action):
        mv_x, mv_y = self.actions[action]
        self.pos = [self.pos[0] + mv_y, self.pos[1] + mv_x]

        # reward = -0.1
        inv_re = -0.01
        if self.pos[0] >= self.c_size:
            self.pos[0] = self.c_size-1
            self.up_mz()
            return self.maze, inv_re, False, {}
        if self.pos[0] < 0:
            self.pos[0] = 0
            self.up_mz()
            return self.maze, inv_re, False, {}
        if self.pos[1] >= self.r_size:
            self.pos[1] = self.r_size-1
            self.up_mz()
            return self.maze, inv_re, False, {}
        if self.pos[1] < 0:
            self.pos[1] = 0
            self.up_mz()
            return self.maze, inv_re, False, {}

        z=0
        if self.pos in self.check_flags:
            self.check_flags.remove(self.pos)
            z = 1

        done = False
        # if self.pos == self.goal:
        if len(self.check_flags)==0:
            lots = len(self.check_flags) / len(self.flags)
            print("## flags={0}/{1}".format(len(self.check_flags), len(self.flags)))
            done = True
            # reward = 10.0
            reward = (1.0 - lots)*10.0
            # reward = 1.0 - lots
        elif z > 0:
            # reward = 0.1
            reward = 1.0
        else:
            # reward = -0.1
            reward = -0.01
            # reward = 0

        # if z>0:
        #     fl_i = self.flg_dict[tuple(self.pos)]
        #     self.flg_data[fl_i] = 0.1

        self.up_mz()
        return self.maze, reward, done, {}

    def reset(self):
        self.pos = np.copy(self.start)
        self.goal = [self.max_index, self.max_index]

        self.maze.fill(self.PATH)
        self.maze[self.start] = self.AGENT
        for fg in self.flags:
            self.maze[fg] = self.FLAG

        self.check_flags = self.flags.copy()
        # self.flg_data = np.ones((len(self.flags),))

        self.st = np.copy(self.maze)
        return self.st

    def up_mz(self):
        self.maze.fill(self.PATH)
        self.maze[self.pos] = self.AGENT
        for fg in self.check_flags:
            self.maze[fg] = self.FLAG


m_size = 30
channel = 4


class MazeProcessor(Processor):
    def __init__(self):
        self.rgb_state = np.zeros((m_size, m_size, channel))

    def process_step(self, observation, reward, done, info):
        old_rgb_state = self.rgb_state.copy()
        self.rgb_state[:, :, 0] = observation.flatten()
        for i in range(1, channel):
            self.rgb_state[:, :, i] = old_rgb_state[:, :, i-1]

        return self.rgb_state, reward, done, info


def main():
    from keras.callbacks import TensorBoard
    env = Mz()
    nb_actions = env.action_space.n
    # n_szie = env.a_self
    # n_szie = 500
    ac = 'relu'
    # ac = 'sigmoid'
    # ac = 'tanh'
    a = 0.24
    # model = Sequential()
    # model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    #
    # model.add(Dense(n_szie))
    # # # # model.add(Dense(n_szie, input_shape=(1, n_szie)))
    # # # # model.add(Activation(ac))
    # model.add(LeakyReLU(alpha=a))
    # # model.add(Dropout(drop))
    # model.add(BatchNormalization())
    #
    # model.add(Dense(n_szie))
    # # model.add(Activation(ac))
    # model.add(LeakyReLU(alpha=a))
    # # model.add(Dropout(drop))
    # model.add(BatchNormalization())
    #
    # model.add(Dense(nb_actions))
    # # model.add(Activation('softmax'))
    # # model.add(Activation('relu'))
    # model.add(LeakyReLU(alpha=a))
    # # model.add(Dense(1, activation='linear'))
    # # model.add(Activation('linear'))
    # print(model.summary())
    n_filters = 32
    kernel = (3, 3)
    strides = (2, 2)

    model = Sequential()
    model.add(Reshape((m_size, m_size, channel), input_shape=(1, m_size, m_size, channel)))
    model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(16, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(nb_actions, activation="relu"))

    processor = MazeProcessor()
    # experience replay用のmemory
    memory = SequentialMemory(limit=5000, window_length=1)
    # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
    policy = EpsGreedyQPolicy(eps=0.1)
    # policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=5000,
                   enable_dueling_network=False, dueling_type='avg', target_model_update=1e-2, policy=policy,
                   processor=processor)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    tb = TensorBoard()
    history = dqn.fit(env, nb_steps=5000000, visualize=False, verbose=2, nb_max_episode_steps=5000,
                      callbacks=[tb])


if __name__ == '__main__':
    main()
    # test()