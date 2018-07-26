# coding=utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
from chainerrl import q_functions
from chainerrl import policies
from chainerrl.agents.ddpg import DDPGModel
from chainer import optimizers
from chainerrl import replay_buffer
from chainerrl.agents.ddpg import DDPG


env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
action_space = env.action_space
action_size = action_space.n

n_hidden_channels = 20
n_hidden_layers = 2
actor_lr = 1e-4
critic_lr = 1e-3

q_func = q_functions.FCSAQFunction(
    obs_size, action_size,
    n_hidden_channels=n_hidden_channels,
    n_hidden_layers=n_hidden_layers)
pi = policies.FCDeterministicPolicy(
    obs_size, action_size=action_size,
    n_hidden_channels=n_hidden_channels,
    n_hidden_layers=n_hidden_layers,
    min_action=0, max_action=1,
    bound_action=True)
model = DDPGModel(q_func=q_func, policy=pi)
opt_a = optimizers.Adam(alpha=actor_lr)
opt_c = optimizers.Adam(alpha=critic_lr)
opt_a.setup(model['policy'])
opt_c.setup(model['q_function'])
opt_a.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
opt_c.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')

gamma = 0.95

def random_action_func():
    return gym.spaces.np_random.uniform(low=0, high=1, size=2).astype(np.float32)


explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3,
    # random_action_func=env.action_space.sample
    random_action_func=random_action_func
)

rbuf = replay_buffer.ReplayBuffer(capacity=10 ** 6)


def phi(obs):
    return obs.astype(np.float32)


replay_start_size = 500
update_interval = 1
target_update_interval = 100
minibatch_size = 64
soft_update_tau = 1e-2
target_update_method = 'hard'
n_update_times = 1
agent = DDPG(model, opt_a, opt_c, rbuf,
             gamma=gamma,
             explorer=explorer,
             replay_start_size=replay_start_size,
             target_update_method=target_update_method,
             target_update_interval=target_update_interval,
             update_interval=update_interval,
             soft_update_tau=soft_update_tau,
             n_times_update=n_update_times,
             phi=phi,
             minibatch_size=minibatch_size)


n_episodes = 500
max_episode_len = 200
for i in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < max_episode_len:
        # Uncomment to watch the behaviour
        # env.render()
        action = agent.act_and_train(obs, reward)
        ai = np.argmax(action)
        obs, reward, done, _ = env.step(ai)
        R += reward
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
    agent.stop_episode_and_train(obs, reward, done)
print('Finished.')

for i in range(10):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < 200:
        env.render()
        action = agent.act(obs)
        ai = np.argmax(action)
        obs, r, done, _ = env.step(ai)
        R += r
        t += 1
    print('test episode:', i, 'R:', R)
    agent.stop_episode()
env.close()