# coding=utf-8

import gym
import gym.spaces
import numpy as np

from mz_data import MzData


class MzEnv(gym.core.Env):
    def __init__(self, name: str):
        self.name = name
        self.AGENT = 0.3
        self.FLAG = 0.7
        self.PATH = 1
        self.WALL = 0

        self.reward_move = -0.01
        self.reward_hit_wall = -1.5
        self.reward_get_flag = 1.5
        self.reward_goal = 10.

        mz_data = MzData()
        mz_data.read(csv=name+'.csv')
        self.flags = mz_data.flags
        self.start = mz_data.start
        self.goal = mz_data.goal
        self.walls = mz_data.walls
        self.maze = np.ones(mz_data.shape)

        self.actions = {
            0: (0, 1),
            1: (0, -1),
            2: (1, 0),
            3: (-1, 0)
        }

        # rigtht, left, up, down,
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.check_flags = self.flags.copy()

        low = self.WALL
        high = self.PATH
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=self.maze.shape, dtype=np.float32)

        self.bl_cnt = 0

    def step(self, action):
        # print("action = ", action)
        mv_x, mv_y = self.actions[action]

        cu_pos = [self.pos[0] + mv_y, self.pos[1] + mv_x]

        if cu_pos in self.walls:
            self.bl_cnt += 1
            self.up_mz()
            return self.maze, self.reward_hit_wall, False, {}

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
            reward = (1.0 - lots)*self.reward_goal
            # reward = 1.0 - lots
        elif z > 0:
            # reward = 0.1
            reward = self.reward_get_flag
        else:
            # reward = -0.1
            reward = self.reward_move
            # reward = 0

        self.up_mz()
        return self.maze, reward, done, {}

    def reset(self):
        print("self.bl_cnt={0}  checked flags={1}/{2}".
              format(self.bl_cnt, len(self.flags)-len(self.check_flags), len(self.flags)))

        self.bl_cnt = 0
        self.pos = np.copy(self.start)
        self.up_mz()
        self.check_flags = self.flags.copy()

        return self.maze

    def up_mz(self):
        self.maze.fill(self.PATH)
        self.maze[tuple(self.pos)] = self.AGENT
        for fg in self.check_flags:
            self.maze[tuple(fg)] = self.FLAG

        # pil_img = Image.fromarray(self.maze*255.).convert("L")
        # pil_img.save('./tmp/lenna_changed_{0}.png'.format(self.cnt))
