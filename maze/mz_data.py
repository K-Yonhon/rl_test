# coding=utf-8

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