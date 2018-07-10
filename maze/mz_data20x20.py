# coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, MaxPooling2D, Permute

from mz_env import MzEnv
from train import train
from my_config import set_tf


def build_model_0(channel, env):
    n_filters = 6
    kernel = (3, 3)
    strides = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters, kernel, padding="same", activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    # model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # model.add(Dense(32, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(nb_actions, activation="relu"))
    return model

def build_model(channel, env):
    n_filters = 10
    kernel = (3, 3)
    strides = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters, kernel, padding="same", activation="relu"))
    # model.add(Conv2D(n_filters, kernel, padding="same", activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    # model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(nb_actions, activation="relu"))
    return model

def main():
    set_tf()

    # flag=33
    csv_name = 'data20x20'
    channel = 4

    env = MzEnv(csv_name)
    model = build_model(channel, env)
    print(model.summary())

    men_limit = 10000
    nb_steps = 500000
    nb_max_episode_steps = 1000
    batch_size = 64
    dqn, history = train(model=model,
                         name=csv_name,
                         env=env,
                         men_limit=men_limit, men_window_length=channel,
                         nb_steps=nb_steps,
                         batch_size=batch_size,
                         nb_max_episode_steps=nb_max_episode_steps)


if __name__ == '__main__':
    main()