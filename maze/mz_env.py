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
        self.MARK = -1.
        self.GOAL = 0.9

        self.reward_move = -0.05
        self.reward_hit_wall = -3.
        self.reward_get_flag = 5.
        self.reward_goal = 10.

        mz_data = MzData()
        mz_data.read(csv=name+'.csv')
        self.flags = mz_data.flags
        self.start = mz_data.start
        self.goal = mz_data.goal
        self.walls = mz_data.walls
        self.maze = np.ones(mz_data.shape)

        # self.move_log = {}
        # self.hit_log = {}
        self.tr = {}
        self.flg_tr = []

        self.actions = {
            0: (0, 1),
            1: (0, -1),
            2: (1, 0),
            3: (-1, 0)
        }

        # rigtht, left, up, down,
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.check_flags = self.flags.copy()

        # low = self.WALL
        low = -1.
        high = self.PATH
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=self.maze.shape, dtype=np.float32)

        self.bl_cnt = 0

    def step(self, action):
        # print("action = ", action)
        mv_x, mv_y = self.actions[action]

        cu_pos = [self.pos[0] + mv_y, self.pos[1] + mv_x]
        # print("self.pos={0}, cu_pos={1}".format(self.pos, cu_pos))
        # cu_pos = [self.pos[0], self.pos[1]]

        # if 0<=cu_pos[0]<self.maze.shape[0] and 0<=cu_pos[1]<self.maze.shape[1]:
        #     if cu_pos[0] == 0 or cu_pos[1] == 0 or cu_pos[0] == self.maze.shape[0]-1 or cu_pos[1] == self.maze.shape[1]-1:
        #         self.bl_cnt += 1
        #         self.pos = [self.pos[0] + mv_y, self.pos[1] + mv_x]
        #         self.up_mz()
        #         return self.maze, self.reward_hit_wall, False, {}

        if cu_pos in self.walls:
            # print("hit pos = ", cu_pos)
            reward_hit = self.reward_hit_wall
            # tp_pos = tuple(cu_pos)
            # if tp_pos in self.hit_log:
            #     self.hit_log[tp_pos] += 1
            #     reward_hit = self.reward_hit_wall - self.hit_log[tp_pos]*0.05
            #     # h_cnt = self.hit_log[tp_pos]
            #     # reward_hit -= 0.05 / ((float(h_cnt) + 0.01) ** 0.5)
            # else:
            #     self.hit_log[tp_pos] = 0
            #     # 0.05 / (0.01 ** 0.5)

            # if tuple(cu_pos) in self.tr:
            #     self.tr[tuple(cu_pos)] += 1
            # else:
            #     self.tr[tuple(cu_pos)] = 1


            self.bl_cnt += 1
            self.up_mz()
            return self.maze, reward_hit, False, {}
        else:
            pass
            # print("not hit pos = ", cu_pos)


        if tuple(cu_pos) in self.tr:
            self.tr[tuple(cu_pos)] += 1
        #     mv_cnt = self.tr[tuple(cu_pos)]
        #     reward += 0.1 / ((float(mv_cnt) + 0.01) ** 0.5)
        else:
            self.tr[tuple(cu_pos)] = 1
        #     reward += 0.1 / (0.01 ** 0.5)


        self.pos = [self.pos[0] + mv_y, self.pos[1] + mv_x]

        z=0
        if self.pos in self.check_flags:
            self.check_flags.remove(self.pos)
            z = 1
            self.flg_tr.append(self.pos)

        done = False
        # if self.pos == self.goal:
        if (len(self.flags)-len(self.check_flags)) >= len(self.flags)*0.8:
        # if len(self.check_flags)==0:
            lots = len(self.check_flags) / len(self.flags)
            print("## DONE self.pos={0}, bl_cnt={1}, flags={2}/{3}".format(self.pos,
                                                                           self.bl_cnt,
                                                                           len(self.flags) - len(self.check_flags),
                                                                           len(self.flags)))
            print("## DONE self.flg_tr={0}".format(self.flg_tr))
            done = True
            # reward = 10.0
            reward = (1.0 - lots)*self.reward_goal
            # reward = 1.0 - lots
        elif z > 0:
            # reward = 0.
            reward = self.reward_get_flag
        else:
            # reward = -0.1
            reward = self.reward_move
            # reward = 0

        # tp_pos = tuple(self.pos)
        # if tp_pos in self.tr:
        #     # self.move_log[tp_pos] += 1
        #     mv_cnt = self.tr[tp_pos]
        #
        #     # reward = self.reward_move - self.move_log[tp_pos]*0.05
        #     reward += 0.01/((float(mv_cnt) + 0.01)**0.5)
        # else:
        #     # self.move_log[tp_pos] = 0
        #     reward += 0.01 / (0.01 ** 0.5)

        self.up_mz()
        return self.maze, reward, done, {}

    def reset(self):
        print("self.bl_cnt={0}  checked flags={1}/{2}".
              format(self.bl_cnt, len(self.flags)-len(self.check_flags), len(self.flags)))

        re_tr = sorted(self.tr.items(), key=lambda x: -x[1])
        print("self.tr={0}".format(re_tr))

        self.bl_cnt = 0
        # self.move_log = {}
        # self.hit_log = {}
        self.tr = {}
        self.flg_tr = []

        self.pos = np.copy(self.start)
        self.check_flags = self.flags.copy()
        self.up_mz()

        return self.maze

    def up_mz(self):
        self.maze.fill(self.PATH)

        # for ps in self.tr:
        #     # c = self.tr[ps]
        #     # if c > 10.:
        #     #     c = 10.
        #     self.maze[ps] = self.MARK

        self.maze[tuple(self.pos)] = self.AGENT
        for fg in self.check_flags:
            self.maze[tuple(fg)] = self.FLAG

        for ps in self.flg_tr:
            # c = self.tr[ps]
            # if c > 10.:
            #     c = 10.
            self.maze[tuple(ps)] = self.MARK

        self.maze[tuple(self.goal)] = self.GOAL
        # pil_img = Image.fromarray(self.maze*255.).convert("L")
        # pil_img.save('./tmp/lenna_changed_{0}.png'.format(self.cnt))
