import numpy as np
import math
import torch
from collections import defaultdict


class Sarsa(object):
    def __init__(self, n_actions, cfg):
        self.n_actions = n_actions
        self.lr = cfg.lr  # 学习率
        self.gamma = cfg.gamma  # 折损因子
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start  # e-greedy策略中epsilon的初始值
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))  # 用嵌套字典存放状态->动作->状态-动作值（Q值）的映射，即Q表

    def sample(self, state):
        """
        采样动作，训练时用
        :param state:
        :return:
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

    def predict(self, state):
        return np.argmax(self.Q_table[str(state)])

    def learn(self, state, action, reward, next_state, next_action, terminated):
        q_predict = self.Q_table[str(state)][action]
        if terminated:
            q_target = reward
        else:
            q_target = reward + self.gamma * self.Q_table[str(next_state)][
                next_action]  # 与Q learning不同，Sarsa是拿下一步动作对应的Q值去更新
        self.Q_table[str(state)][action] += self.lr * (q_target - q_predict)

    def save(self, path):
        """
        将Q表格保存到文件
        :param path:
        :return:
        """
        import dill
        torch.save(
            obj=self.Q_table,
            f=path + 'sarsa_model.pkl',
            pickle_module=dill
        )

    def load(self, path):
        '''从文件中读取数据到 Q表格
        '''
        import dill
        self.Q_table = torch.load(
            f=path + 'sarsa_model.pkl',
            pickle_module=dill
        )
