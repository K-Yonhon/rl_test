# coding=utf-8

import gym
import gym.spaces
import numpy as np
import os
import random as rn
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, MaxPooling2D, Permute
# from operator import truediv
from PIL import Image, ImageDraw

from mz_data import MzData

import tensorflow as tf
from keras.backend import tensorflow_backend, set_session

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)
set_session(tf.Session(config=config))

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)

csv_name = 'data20x20'


class Mz(gym.core.Env):
    def __init__(self):
        self.AGENT = 0.3
        self.FLAG = 0.7
        self.PATH = 1
        self.WALL = 0

        mz_data = MzData()
        # data20x20.csv f=33
        mz_data.read(csv=csv_name+'.csv')
        self.flags = mz_data.flags
        self.c_size = mz_data.shape[1]
        self.r_size = mz_data.shape[0]
        self.start = mz_data.start
        self.goal = mz_data.goal
        self.walls = mz_data.walls

        self.maze = np.ones(mz_data.shape)

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

        self.flg_dict = {}
        for i, flg_pos in enumerate(self.flags):
            self.flg_dict[tuple(flg_pos)] = i

        low = self.WALL
        high = self.PATH
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=self.maze.shape, dtype=np.float32)

        self.bl_cnt = 0

    def step(self, action):
        # print("action = ", action)
        mv_x, mv_y = self.actions[action]

        cu_pos = [self.pos[0] + mv_y, self.pos[1] + mv_x]

        inv_re = -1.5
        if cu_pos in self.walls:
            self.bl_cnt += 1
            self.up_mz()
            return self.maze, inv_re, False, {}

        self.pos = [self.pos[0] + mv_y, self.pos[1] + mv_x]

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
            reward = 1.5
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
        print("self.bl_cnt={0}  checked flags={1}/{2}".
              format(self.bl_cnt, len(self.flags)-len(self.check_flags), len(self.flags)))

        self.cnt = 0
        self.bl_cnt = 0

        self.pos = np.copy(self.start)
        self.goal = [self.max_index, self.max_index]

        self.up_mz()

        self.check_flags = self.flags.copy()

        self.st = np.copy(self.maze)
        return self.st

    def up_mz(self):
        self.maze.fill(self.PATH)
        for wall in self.walls:
            self.maze[tuple(wall)] = self.WALL

        self.maze[tuple(self.pos)] = self.AGENT

        for fg in self.check_flags:
            self.maze[tuple(fg)] = self.FLAG

        # pil_img = Image.fromarray(self.maze*255.).convert("L")
        # pil_img.save('./tmp/lenna_changed_{0}.png'.format(self.cnt))

        self.cnt += 1


# m_size = 10
channel = 4


def main():
    from keras.callbacks import TensorBoard
    env = Mz()
    env.seed(7)
    nb_actions = env.action_space.n
    m_size = env.c_size

    n_filters = 4
    kernel = (3, 3)
    strides = (1, 1)

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(channel, m_size, m_size)))
    model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters, kernel, padding="same", activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    # model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # model.add(Dense(32, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(nb_actions, activation="relu"))
    print(model.summary())

    # experience replay用のmemory
    memory = SequentialMemory(limit=5000, window_length=channel)
    # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
    policy = EpsGreedyQPolicy(eps=0.2)
    # policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2,
                   policy=policy)
    dqn.compile(Adam(lr=1e-3, clipnorm=1.), metrics=['mae'])

    tb = TensorBoard(log_dir='./logs/' + csv_name)
    history = dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2, nb_max_episode_steps=1000,
                      callbacks=[tb])


if __name__ == '__main__':
    main()