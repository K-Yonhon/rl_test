# coding=utf-8

import gym
import gym.spaces
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from keras.models import load_model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.agents import DDPGAgent
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd


class MzData:
    def read(self, csv: str, start_symbol: str="s", goal_symbol: str="g", flag_symbol: str="f"):
        df = pd.read_csv(csv, dtype='str', index_col='row')
        self.shape = df.shape

        df_fg = df == flag_symbol
        df_start = df == start_symbol
        df_goal = df == goal_symbol
        self.flags = []

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                if df_fg.iat[row, col]:
                    self.flags.append([row, col])
                if df_start.iat[row, col]:
                    self.start = [row, col]
                if df_goal.iat[row, col]:
                    self.goal = [row, col]


class Mz(gym.core.Env):
    def __init__(self):
        mz_data = MzData()
        # mz_data.read(csv='data10x10.csv')
        mz_data.read(csv='data30x30.csv')

        self.c_flags = mz_data.flags
        self.c_size = mz_data.shape[1]
        self.r_size = mz_data.shape[0]
        self.start = mz_data.start
        self.goal = mz_data.goal

        self.pos = [0, 0]
        self.st = None
        self.actions = {
            0: (0, 1),
            1: (0, -1),
            2: (1, 0),
            3: (-1, 0)
        }
        self.my = 0.2
        self.remark = 0.65
        # rigtht, left, up, down,
        self.action_space = gym.spaces.Discrete(4)

        self.a_self = self.c_size*self.r_size
        low = np.zeros((self.a_self, ))
        high = np.ones((self.a_self,))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        mv_x, mv_y = self.actions[action]
        cuy, cux = self.pos[0] + mv_y, self.pos[1] + mv_x

        inv_re = -0.01
        # inv_re = -0.005
        if cux >= self.c_size:
            return self.st,inv_re, False, {}
        if cux < 0:
            return self.st, inv_re, False, {}
        if cuy >= self.r_size:
            return self.st, inv_re, False, {}
        if cuy < 0:
            return self.st, inv_re, False, {}

        if [cuy, cux] in self.wall:
            return self.st, inv_re, False, {}

        self.pos = [self.pos[0] + mv_y, self.pos[1] + mv_x]

        z=0
        if self.pos in self.flags:
            self.flags.remove(self.pos)
            # index = self.pos[0] + self.pos[1] * self.r_size
            # self.st[index] = self.remark
            # self.st[index] = 1.0
            z = 1

        done = False
        # if self.pos == [self.c_size - 1, self.r_size - 1]:
        if self.pos == self.goal:
            lots = len(self.flags) / len(self.c_flags)
            print("## flags={0}/{1}, pos={2}".format(len(self.flags), len(self.c_flags), self.flags))
            done = True
            # reward = 10.0
            reward = (1.0 - lots)*100.0
        elif z > 0:
            # reward = 10.0
            reward = 5.0
        else:
            # reward = -0.1
            reward = -0.01
            # reward = 0

        self.st[:] = 1.0
        for y, x in self.wall:
            index = x + y * self.r_size
            self.st[index] = 0.0

        for y, x in self.flags:
            index = x + y * self.r_size
            self.st[index] = 0.5

        index = self.pos[1] + self.pos[0] * self.r_size
        self.st[index] = self.my

        return self.st, reward, done, {}

    def reset(self):
        self.pos = self.start
        self.st = np.ones((self.a_self,), dtype=np.float32)
        # self.goal = self.goal
        # self.st[2] = self.remark
        # self.st[:self.a_self] = self.remark
        # self.wall = [[2,0], [2,1], [2,1], [2,1], [2,1],
        #              [7,5], [7,6], [7,7], [7,8], [7,9]]
        # self.wall = [[2,0], [2,1], [2,1], [2,1], [2,1]]
        self.wall = []
        for y, x in self.wall:
            index = x + y * self.r_size
            self.st[index] = 0.0

        self.flags = self.c_flags.copy()

        for y, x in self.flags:
            index = x + y * self.r_size
            self.st[index] = 0.5

        self.st[0] = self.my

        return self.st


def main():
    from keras.callbacks import TensorBoard

    env = Mz()
    nb_actions = env.action_space.n
    n_szie = env.a_self
    # n_szie = 500
    # ac = 'relu'
    ac = 'sigmoid'
    ac = 'tanh'
    a = 0.24
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))

    model.add(Dense(n_szie))
    model.add(LeakyReLU(alpha=a))

    model.add(Dense(n_szie))
    model.add(LeakyReLU(alpha=a))

    model.add(Dense(n_szie))
    model.add(LeakyReLU(alpha=a))

    model.add(Dense(nb_actions))
    model.add(LeakyReLU(alpha=a))
    # model.add(Dense(1, activation='linear'))
    print(model.summary())

    # experience replay用のmemory
    memory = SequentialMemory(limit=5000, window_length=1)
    # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
    policy = EpsGreedyQPolicy(eps=0.3)
    # policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=5000,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    tb = TensorBoard()
    history = dqn.fit(env, nb_steps=5000000, visualize=False, verbose=2, nb_max_episode_steps=5000,
                      callbacks=[tb])

    # dqn.save_weights('dqn_weights.h5f', overwrite=True)
    print('save the architecture of a model')
    json_string = model.to_json()
    open('model.json', 'w').write(json_string)

    print('save weights')
    model.save_weights('model_weights.hdf5')

    # policy = BoltzmannQPolicy()
    # sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=100, policy=policy)
    # sarsa.compile(Adam(lr=1e-3), metrics=['mae'])
    #
    # # Okay, now it's time to learn something! We visualize the training here for show, but this
    # # slows down training quite a lot. You can always safely abort the training prematurely using
    # # Ctrl + C.
    # sarsa.fit(env, nb_steps=50000, visualize=False, verbose=2)
    #
    # # After training is done, we save the final weights.
    # sarsa.save_weights('sarsa_mz_weights.h5f', overwrite=True)
    #
    # # Finally, evaluate our algorithm for 5 episodes.
    # sarsa.test(env, nb_episodes=5, visualize=True)


def test():
    model_filename = 'model.json'
    weights_filename = 'model_weights.hdf5'
    # model = load_model(weights_filename, compile=False)
    json_string = open(model_filename).read()
    model = model_from_json(json_string)
    model.load_weights(weights_filename)
    # print(model.summary())

    env = Mz()
    env.reset()

    while True:
        state = env.st
        ss = np.reshape(state, (1, 1, 100))
        # state = np.array([[0, 0, 0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
        # ss = np.reshape(state, ())
        actions = model.predict(ss)
        action = np.argmax(actions)
        print("action=", action)
        st, reward, done, _ = env.step(action)
        if done:
            print("##done")
            break

    # # dqn.load_weights(weights_filename)


if __name__ == '__main__':
    main()
    # test()