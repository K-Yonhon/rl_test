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

def build_model_1(env):
    channel = 4
    n_filters1 = 32
    n_filters2 = 16
    n_filters3 = 16
    kernel = (3, 3)
    strides = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(n_filters1, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters2, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters3, kernel, padding="same", activation="relu"))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_actions, activation="relu"))

    men_limit = 10000
    batch_size = 64
    nb_steps_warmup = 500
    return model, channel, men_limit, batch_size, nb_steps_warmup


def build_model_2(env):
    channel = 4

    n_filters1 = 32
    n_filters2 = 16
    n_filters3 = 16
    kernel = (3, 3)
    strides = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(n_filters1, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters2, kernel, strides=strides, padding="same", activation="relu"))
    model.add(Conv2D(n_filters3, kernel, padding="same", activation="relu"))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_actions, activation="relu"))

    men_limit = 50000
    batch_size = 64
    nb_steps_warmup = 500
    return model, channel, men_limit, batch_size, nb_steps_warmup


def build_model_3(env):
    channel = 4

    n_filters1 = 32
    n_filters2 = 16
    n_filters3 = 16
    kernel = (3, 3)
    strides = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(n_filters1, kernel, strides=strides, padding="same", activation="relu", kernel_initializer="he_normal"))
    model.add(Conv2D(n_filters2, kernel, strides=strides, padding="same", activation="relu", kernel_initializer="he_normal"))
    model.add(Conv2D(n_filters3, kernel, padding="same", activation="relu", kernel_initializer="he_normal"))

    model.add(Flatten())

    model.add(Dense(512, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(nb_actions, activation="relu", kernel_initializer="he_normal"))

    men_limit = 10000
    batch_size = 64
    nb_steps_warmup = 500
    return model, channel, men_limit, batch_size, nb_steps_warmup


def build_model_4(env):
    channel = 4

    n_filters1 = 8
    n_filters2 = 4
    n_filters3 = 4
    kernel = (3, 3)
    strides = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(n_filters1, kernel, strides=strides, padding="same", activation="relu", kernel_initializer="he_normal"))
    model.add(Conv2D(n_filters2, kernel, strides=strides, padding="same", activation="relu", kernel_initializer="he_normal"))
    model.add(Conv2D(n_filters3, kernel, padding="same", activation="relu", kernel_initializer="he_normal"))

    model.add(Flatten())

    model.add(Dense(512, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(nb_actions, activation="relu", kernel_initializer="he_normal"))

    men_limit = 20000
    batch_size = 64
    nb_steps_warmup = 500
    return model, channel, men_limit, batch_size, nb_steps_warmup


def build_model_5(env):
    channel = 4

    n_filters1 = 8
    n_filters2 = 4
    n_filters3 = 4
    kernel = (3, 3)
    strides = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(
        Conv2D(n_filters1, kernel, strides=strides, padding="same", activation="relu", kernel_initializer="he_normal"))
    model.add(
        Conv2D(n_filters2, kernel, strides=strides, padding="same", activation="relu", kernel_initializer="he_normal"))
    model.add(Conv2D(n_filters3, kernel, padding="same", activation="relu", kernel_initializer="he_normal"))

    model.add(Flatten())

    model.add(Dense(512, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(nb_actions, activation="relu", kernel_initializer="he_normal"))

    men_limit = 5000
    batch_size = 64
    nb_steps_warmup = 500
    return model, channel, men_limit, batch_size, nb_steps_warmup


def build_model_6(env):
    channel = 4

    n_filters1 = 8
    n_filters2 = 4
    n_filters3 = 4
    kernel = (3, 3)
    strides = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(
        Conv2D(n_filters1, kernel, strides=strides, padding="same", activation="relu", kernel_initializer="he_normal"))
    model.add(
        Conv2D(n_filters2, kernel, strides=strides, padding="same", activation="relu", kernel_initializer="he_normal"))
    model.add(Conv2D(n_filters3, kernel, padding="same", activation="relu", kernel_initializer="he_normal"))

    model.add(Flatten())

    model.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(nb_actions, activation="relu", kernel_initializer="he_normal"))

    men_limit = 5000
    batch_size = 64
    nb_steps_warmup = 500
    return model, channel, men_limit, batch_size, nb_steps_warmup

def build_model(env):
    channel = 8

    n_filters1 = 16
    n_filters2 = 8
    n_filters3 = 8
    kernel = (3, 3)
    strides = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(
        Conv2D(n_filters1, kernel, strides=strides, padding="same", activation="relu", kernel_initializer="he_normal"))
    model.add(
        Conv2D(n_filters2, kernel, strides=strides, padding="same", activation="relu", kernel_initializer="he_normal"))
    model.add(Conv2D(n_filters3, kernel, padding="same", activation="relu", kernel_initializer="he_normal"))

    model.add(Flatten())

    model.add(Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(nb_actions, activation="relu", kernel_initializer="he_normal"))

    men_limit = 5000
    batch_size = 64
    nb_steps_warmup = 500
    return model, channel, men_limit, batch_size, nb_steps_warmup

def main():
    set_tf()

    # flag=33
    csv_name = 'data20x20'
    # channel = 4

    env = MzEnv(csv_name)
    model, channel, men_limit, batch_size, nb_steps_warmup = build_model(env)
    print(model.summary())

    # men_limit = 10000
    nb_steps = 500000
    nb_max_episode_steps = 1600
    # nb_steps_warmup = 500
    # batch_size = 64
    dqn, history = train(model=model,
                         name=csv_name,
                         env=env,
                         men_limit=men_limit, men_window_length=channel,
                         nb_steps=nb_steps,
                         batch_size=batch_size,
                         nb_max_episode_steps=nb_max_episode_steps,
                         nb_steps_warmup=nb_steps_warmup)


if __name__ == '__main__':
    main()