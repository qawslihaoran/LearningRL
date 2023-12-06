import datetime
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch


class Config:
    """
    配置
    """

    def __init__(self):
        self.env_name = 'CliffWalking-v0'  # 环境名称
        self.algo_name = 'Q-Learning'  # 算法名称
        self.train_eps = 400  # 训练回合数
        self.test_eps = 10  # 测试回合数
        self.max_steps = 200  # 单回合最大步数
        self.epsilon_start = 0.95  # e-greedy策略中epsilon的初始值
        self.epsilon_end = 0.01  # e-greedy策略中epsilon的最终值
        self.epsilon_decay = 300  # e-greedy策略中epsilon的衰减率
        self.gamma = 0.9  # 折损因子
        self.lr = 0.1  # 学习率
        self.seed = 1  # 随机数种子
        if torch.cuda.is_available():  # 是否使用GPUs
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')


def smooth(data, weight=0.9):
    """
    用于平滑曲线
    :param data:
    :param weight:
    :return:
    """
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards(rewards, title="learning curve"):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{title}")
    plt.xlim(0, len(rewards), 10)  # 设置x轴的范围
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()

