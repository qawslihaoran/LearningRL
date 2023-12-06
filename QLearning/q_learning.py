import numpy as np
import math
import torch
from collections import defaultdict
import config


class QLearning(object):
    def __init__(self, n_states, n_actions, cfg):
        self.n_actions = n_actions
        self.lr = cfg.lr  # 学习率
        self.gamma = cfg.gamma  # 折损因子
        self.sample_count = 0
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start  # e-greedy策略中epsilon的初始值
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))  # 用嵌套字典存放状态->动作->状态-动作值（Q值）的映射，即Q表

    def sample_action(self, state):
        """
        采样动作，训练时用
        :param state:
        :return: action, 返回动作
        """
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.sample_count / self.epsilon_decay)  # epsilon是会递减的，这里选择指数递减
        # e-greedy 策略
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.n_actions)
        return action

    def predict_action(self, state):
        """
        预测或选择动作，测试时用
        :param state:
        :return:
        """
        action = np.argmax(self.Q_table[str(state)])
        return action

    def learn(self, state, action, reward, next_state, terminated):
        q_predict = self.Q_table[str(state)][action]
        if terminated:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.lr * (q_target - q_predict)

