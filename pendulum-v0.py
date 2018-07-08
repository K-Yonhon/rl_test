# coding=utf-8

import gym
from rl.core import Processor
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.callbacks import TensorBoard


# GymのPendulum環境を作成
env = gym.make("Pendulum-v0")

# 取りうる”打ち手”のアクション数と値の定義
nb_actions = 2
ACT_ID_TO_VALUE = {0: [-1], 1: [+1]}


class PendulumProcessor(Processor):

    # Duel-DQNの出力と、Gym環境の入力の違いを吸収
    def process_action(self, action):
        return ACT_ID_TO_VALUE[action]

    # Gym環境の報酬の出力と、Duel-DQNの報酬の入力との違いを吸収
    # def process_reward(self, reward):
    #     if reward > -0.2:
    #         return 1
    #     elif reward > -1.0:
    #         return 0
    #     else:
    #         return 0
    def process_reward(self, reward):
        if reward > -0.2:
            return 3
        elif reward > -1.0:
            return 0.
        else:
            return reward/15.

    def process_state_batch(self, batch):
        # processed_batch = batch.astype('float32') / 255.
        return batch
        # return np.zeros((1,128,128,3))

processor = PendulumProcessor()

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dense(nb_actions, activation="linear"))

# Duel-DQNアルゴリズム関連の幾つかの設定
memory = SequentialMemory(limit=10000, window_length=1)
# policy = BoltzmannQPolicy()
policy = EpsGreedyQPolicy(eps=0.2)

# Duel-DQNのAgentクラスオブジェクトの準備 （上記processorやmodelを元に）
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               enable_dueling_network=True, dueling_type="avg", target_model_update=1e-2, policy=policy,
               processor=processor)
dqn.compile(Adam(lr=1e-3), metrics=["mae"])
print(dqn.model.summary())

tb = TensorBoard()
# 定義課題環境に対して、アルゴリズムの学習を実行 （必要に応じて適切なCallbackも定義、設定可能）
# 上記Processorクラスの適切な設定によって、Agent-環境間の入出力を通して設計課題に対しての学習が進行
dqn.fit(env, nb_steps=100000, visualize=True, verbose=2, callbacks=[tb])

# 学習後のモデルの重みの出力
dqn.save_weights("duel_dqn_{}_weights.h5f".format("Pendulum-v0"), overwrite=True)

# 学習済モデルに対して、テストを実行 （必要に応じて適切なCallbackも定義、設定可能）
dqn.test(env, nb_episodes=100, visualize=True)