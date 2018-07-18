# coding=utf-8

import os
import random as rn
import numpy as np
import gym
import gym.spaces
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from keras.callbacks import TensorBoard


def train(model,
          name: str,
          env: gym.core.Env,
          men_limit: int,
          men_window_length: int,
          nb_steps: int, nb_max_episode_steps: int,
          nb_steps_warmup=10,
          batch_size: int=32,
          target_model_update=1e-2,
          seed=7,
          visualize=False):

    if seed is not None:
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(seed)
        rn.seed(seed)
        env.seed(seed)

    nb_actions = env.action_space.n
    # experience replay用のmemory
    memory = SequentialMemory(limit=men_limit, window_length=men_window_length)
    # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
    # policy = EpsGreedyQPolicy(eps=0.3)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   nb_steps_warmup=nb_steps_warmup,
                   batch_size=batch_size,
                   enable_dueling_network=True,
                   dueling_type='avg',
                   target_model_update=target_model_update,
                   policy=policy)
    # dqn.compile(Adam(lr=1e-3, clipnorm=1.), metrics=['mae'])
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    tb = TensorBoard(log_dir='./logs/' + name)
    history = dqn.fit(env, nb_steps=nb_steps, visualize=visualize, verbose=2,
                      nb_max_episode_steps=nb_max_episode_steps,
                      callbacks=[tb])
    return dqn, history