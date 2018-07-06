# coding=utf-8

import os
import random as rn
import gym
from rl.core import Processor
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, MaxPooling2D, Permute, Dropout, BatchNormalization
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from PIL import Image, ImageDraw
import numpy as np
from keras.callbacks import TensorBoard

import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

# GymのPendulum環境を作成
env = gym.make("Pendulum-v0")

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)
env.seed(7)

# 取りうる”打ち手”のアクション数と値の定義
nb_actions = 2
ACT_ID_TO_VALUE = {0: [-1], 1: [+1]}

# img_size = 128
img_size = 32
channel = 4


class PendulumProcessor(Processor):
    def __init__(self):
        self.rgb_state = np.zeros((img_size, img_size, channel))
        self.cnt = 0

    # Duel-DQNの出力と、Gym環境の入力の違いを吸収
    def process_action(self, action):
        return ACT_ID_TO_VALUE[action]

    # Gym環境の報酬の出力と、Duel-DQNの報酬の入力との違いを吸収
    def process_reward(self, reward):
        if reward > -0.2:
            return 10
        elif reward > -1.0:
            return 0
        else:
            return reward/10.

    # 状態（x,y座標）から対応画像を描画する関数
    def _get_rgb_state(self, state):

        h_size = img_size/2.0

        img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        dr = ImageDraw.Draw(img)

        # 棒の長さ
        l = img_size/4.0 * 3.0/ 1.5

        # 棒のラインの描写
        dr.line(((h_size - l * state[1], h_size - l * state[0]), (h_size, h_size)),
                (0, 0, 0),
                1)

        # 棒の中心の円を描写（それっぽくしてみた）
        # buff = img_size/32.0
        buff = img_size /(img_size/2.0)
        dr.ellipse(((h_size - buff, h_size - buff), (h_size + buff, h_size + buff)),
                   outline=(0, 0, 0), fill=(255, 0, 0))

        # 画像の一次元化（GrayScale化）とarrayへの変換
        pilImg = img.convert("L")

        # pilImg.save('./tmp/lenna_changed_{0}.png'.format(self.cnt))
        # self.cnt+=1

        img_arr = np.asarray(pilImg)

        # 画像の規格化
        img_arr = img_arr/255.0

        return img_arr

    # Gym環境の出力と、Duel-DQNアルゴリズムへの入力との違いを吸収
    def process_step(self, observation, reward, done, info):
        # old_rgb_state = self.rgb_state.copy()

        # アルゴリズムの状態入力として、画像を用いる（過去３フレームを入力する）
        # 直近の状態に対応する画像を作成
        # self.rgb_state[:, :, 0] = self._get_rgb_state(observation)
        # # 過去２フレームも保持
        # for i in range(1, channel):
        #     self.rgb_state[:, :, i] = np.copy(old_rgb_state[:, :, i-1]) # shift old state
        # for i in range(1, channel):
        #     self.rgb_state[:, :, i-1] = old_rgb_state[:, :, i] # shift old state
        # self.rgb_state[:, :, channel-1] = self._get_rgb_state(observation)

        # アルゴリズムへの報酬として、設定課題に沿った報酬を用いる（上記通り）
        # reward = self.process_reward(reward)
        # return self.rgb_state, reward, done, info
        reward = self.process_reward(reward)
        # return self._get_rgb_state(observation), reward, done, info
        return self._get_rgb_state(observation), reward, done, info

    def process_observation(self, observation):
        # old_rgb_state = self.rgb_state.copy()

        # self.rgb_state[:, :, 0] = self._get_rgb_state(observation)
        # # 過去２フレームも保持
        # for i in range(1, channel):
        #     self.rgb_state[:, :, i] = np.copy(old_rgb_state[:, :, i-1]) # shift old state

        # for i in range(1, channel):
        #     self.rgb_state[:, :, i - 1] = old_rgb_state[:, :, i]  # shift old state
        # self.rgb_state[:, :, channel - 1] = self._get_rgb_state(observation)
        # return self.rgb_state
        return self._get_rgb_state(observation)

    # def process_state_batch(self, batch):
    #     # processed_batch = batch.astype('float32') / 255.
    #     # return processed_batch
    #     return batch


processor = PendulumProcessor()

# 画像の特徴量抽出ネットワークのパラメタ
n_filters = 32
# n_filters = 16
kernel = (3, 3)
strides = (2, 2)

# model = Sequential()
# # 畳込み層による画像の特徴量抽出ネットワーク
# # model.add(Reshape((img_size, img_size, channel), input_shape=(1, img_size, img_size, channel)))
# model.add(Permute((2, 3, 1), input_shape=(channel, img_size, img_size)))
# model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
# model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
# model.add(Conv2D(n_filters, kernel, padding="same", activation="relu"))
# # model.add(Dropout(0.2))
# model.add(Conv2D(n_filters, kernel, padding="same", activation="relu"))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# # 以前と同様の2層FCのQ関数ネットワーク
# # model.add(Dense(16, activation="relu"))
# # model.add(Dropout(0.2))
# # model.add(Dense(16, activation="relu"))
# model.add(Dense(16, activation="relu"))
# model.add(Dense(nb_actions, activation="linear"))

model = Sequential()
# 畳込み層による画像の特徴量抽出ネットワーク
# model.add(Reshape((img_size, img_size, channel), input_shape=(1, img_size, img_size, channel)))
model.add(Permute((2, 3, 1), input_shape=(channel, img_size, img_size)))

model.add(Conv2D(n_filters, kernel, strides=strides, padding="same"))
# model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(n_filters, kernel, strides=strides, padding="same"))
# model.add(BatchNormalization())
model.add(Activation('relu'))
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(n_filters, kernel, strides=strides, padding="same", activation="relu"))
# model.add(Conv2D(n_filters, kernel, padding="same", activation="relu"))
# model.add(Dropout(0.2))
model.add(Conv2D(n_filters, kernel, padding="same"))
# model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
# 以前と同様の2層FCのQ関数ネットワーク
# model.add(Dense(16, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation="relu"))
model.add(Dense(32))
# model.add(Dense(16, use_bias=False))
# model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(16))
# model.add(Dense(16, use_bias=False))
# model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(Dense(nb_actions, activation="linear"))
model.add(Dense(nb_actions))
# model.add(Dense(nb_actions, use_bias=False))
# model.add(BatchNormalization())
model.add(Activation('linear'))

# Duel-DQNアルゴリズム関連の幾つかの設定
memory = SequentialMemory(limit=10000, window_length=channel)
policy = BoltzmannQPolicy()
# policy = EpsGreedyQPolicy(eps=0.2)

# Duel-DQNのAgentクラスオブジェクトの準備 （上記processorやmodelを元に）
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               enable_dueling_network=True,
               dueling_type="avg",
               # dueling_type="max",
               target_model_update=1e-2,
               # batch_size=32,
               policy=policy,
               processor=processor)
dqn.compile(Adam(lr=1e-3, clipnorm=1.), metrics=["mae"])
print(dqn.model.summary())

# dqn.load_weights("duel_dqn_Pendulum-v0_cnn_weights.h5f")

tb = TensorBoard(log_dir='./logs')
# 定義課題環境に対して、アルゴリズムの学習を実行 （必要に応じて適切なCallbackも定義、設定可能）
# 上記Processorクラスの適切な設定によって、Agent-環境間の入出力を通して設計課題に対しての学習が進行
dqn.fit(env, nb_steps=100000,
        # nb_max_episode_steps=300,
        visualize=True, verbose=2, callbacks=[tb])

json_string = model.to_json()
open('cnn_model.json', 'w').write(json_string)

# 学習後のモデルの重みの出力
dqn.save_weights("duel_dqn_{}_weights.h5f".format("Pendulum-v0_cnn"), overwrite=True)

# 学習済モデルに対して、テストを実行 （必要に応じて適切なCallbackも定義、設定可能）
dqn.test(env, nb_episodes=100, visualize=True)
