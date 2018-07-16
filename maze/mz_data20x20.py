# coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, MaxPooling2D, Permute, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM

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

def build_model_7(env):
    channel = 4

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
    model.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(nb_actions, activation="linear", kernel_initializer="he_normal"))

    men_limit = 5000
    batch_size = 32
    nb_steps_warmup = 200
    return model, channel, men_limit, batch_size, nb_steps_warmup

def build_model_9(env):
    # channel = 32
    channel = 4

    n_filters1 = 16
    kernel1 = (3, 3)

    n_filters2 = 32
    kernel2 = (3, 3)

    n_filters3 = 32
    kernel3 = (2, 2)

    # kernel = (3, 3)
    strides1 = (3, 3)
    strides2 = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Conv2D(n_filters1, kernel1, strides=strides1, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(Dropout(0.2))

    model.add(Conv2D(n_filters2, kernel2, strides=strides2, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(Dropout(0.2))

    model.add(Conv2D(n_filters3, kernel3, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(Dropout(0.2))

    model.add(Flatten())

    # model.add(LSTM(512, activation='tanh'))

    # model.add(Dense(128, kernel_initializer="he_normal", use_bias=False))
    # model.add(BatchNormalization())
    model.add(Dense(256))
    model.add(Activation("relu"))
    # model.add(Dropout(0.2))

    # model.add(Dense(64, kernel_initializer="he_normal", use_bias=False))
    # model.add(BatchNormalization())
    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dense(128))
    model.add(Activation("relu"))

    # model.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
    # model.add(Dense(nb_actions, activation="linear", kernel_initializer="he_normal"))
    model.add(Dense(nb_actions, activation="softmax"))
    # model.add(Dense(nb_actions, activation="linear"))

    men_limit = 20000
    batch_size = 32
    nb_steps_warmup = 200
    # target_model_update = 1000
    target_model_update = 1e-2
    return model, channel, men_limit, batch_size, nb_steps_warmup, target_model_update

def build_model_10(env):
    # channel = 32
    channel = 4

    n_filters1 = 16
    kernel1 = (3, 3)

    n_filters2 = 32
    kernel2 = (3, 3)

    n_filters3 = 32
    kernel3 = (2, 2)

    # kernel = (3, 3)
    strides1 = (3, 3)
    strides2 = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Conv2D(n_filters1, kernel1, strides=strides1, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Conv2D(n_filters2, kernel2, strides=strides2, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Conv2D(n_filters3, kernel3, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Flatten())

    # model.add(LSTM(512, activation='tanh'))

    # model.add(Dense(128, kernel_initializer="he_normal", use_bias=False))
    # model.add(BatchNormalization())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    # model.add(Dense(64, kernel_initializer="he_normal", use_bias=False))
    # model.add(BatchNormalization())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    # model.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
    # model.add(Dense(nb_actions, activation="linear", kernel_initializer="he_normal"))
    model.add(Dense(nb_actions, activation="softmax"))
    # model.add(Dense(nb_actions, activation="linear"))

    men_limit = 20000
    batch_size = 32
    nb_steps_warmup = 200
    # target_model_update = 1000
    target_model_update = 1e-2
    return model, channel, men_limit, batch_size, nb_steps_warmup, target_model_update

def build_model_11(env):
    # channel = 32
    channel = 1

    n_filters1 = 32
    kernel1 = (3, 3)

    n_filters2 = 16
    kernel2 = (3, 3)

    n_filters3 = 16
    kernel3 = (2, 2)

    # kernel = (3, 3)
    strides1 = (3, 3)
    strides2 = (2, 2)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Conv2D(n_filters1, kernel1, strides=strides1, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(Dropout(0.3))

    model.add(Conv2D(n_filters2, kernel2, strides=strides2, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(Dropout(0.3))

    model.add(Conv2D(n_filters3, kernel3, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(Dropout(0.3))

    model.add(Flatten())

    # model.add(LSTM(512, activation='tanh'))

    # model.add(Dense(128, kernel_initializer="he_normal", use_bias=False))
    # model.add(BatchNormalization())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(64))
    model.add(Activation("relu"))
    # model.add(Dropout(0.2))

    # model.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
    # model.add(Dense(nb_actions, activation="linear", kernel_initializer="he_normal"))
    # model.add(Dense(nb_actions, activation="softmax"))
    model.add(Dense(nb_actions, activation="softmax"))

    men_limit = 20000
    batch_size = 64
    nb_steps_warmup = 200
    # target_model_update = 1000
    target_model_update = 1e-3
    return model, channel, men_limit, batch_size, nb_steps_warmup, target_model_update

def build_model(env):
    # channel = 32
    channel = 1

    n_filters1 = 8
    kernel1 = (8, 8)

    n_filters2 = 16
    kernel2 = (4, 4)

    n_filters3 = 32
    kernel3 = (2, 2)

    # kernel = (3, 3)
    strides1 = (2, 2)
    strides2 = (1, 1)

    input_shape = (channel,) + env.observation_space.shape
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Conv2D(n_filters1, kernel1, strides=strides1, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(Dropout(0.3))

    model.add(Conv2D(n_filters3, kernel3, padding="same"))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(Dropout(0.3))

    model.add(Flatten())

    # model.add(LSTM(512, activation='tanh'))

    # model.add(Dense(128, kernel_initializer="he_normal", use_bias=False))
    # model.add(BatchNormalization())
    model.add(Dense(256))
    model.add(Activation("relu"))
    # model.add(Dropout(0.2))

    # model.add(Dense(64))
    # model.add(Activation("relu"))
    # model.add(Dropout(0.2))

    # model.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
    # model.add(Dense(nb_actions, activation="linear", kernel_initializer="he_normal"))
    # model.add(Dense(nb_actions, activation="linear"))
    model.add(Dense(nb_actions, activation="softmax"))

    men_limit = 20000
    batch_size = 32
    nb_steps_warmup = 200
    # target_model_update = 1000
    target_model_update = 1e-3
    return model, channel, men_limit, batch_size, nb_steps_warmup, target_model_update

def main():
    set_tf()

    # flag=33
    csv_name = 'data20x20'
    # channel = 4
    # target_model_update = 1e-2

    env = MzEnv(csv_name)
    model, channel, men_limit, batch_size, nb_steps_warmup, target_model_update = build_model(env)
    print(model.summary())

    # men_limit = 10000
    nb_steps = 1000000
    # nb_max_episode_steps = 1600
    nb_max_episode_steps = 1000
    # nb_steps_warmup = 500
    # batch_size = 64
    dqn, history = train(model=model,
                         name=csv_name,
                         env=env,
                         men_limit=men_limit, men_window_length=channel,
                         nb_steps=nb_steps,
                         batch_size=batch_size,
                         nb_max_episode_steps=nb_max_episode_steps,
                         nb_steps_warmup=nb_steps_warmup,
                         target_model_update=target_model_update)


if __name__ == '__main__':
    main()