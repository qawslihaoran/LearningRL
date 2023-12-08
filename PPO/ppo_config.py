import torch


class Config:
    def __init__(self):
        self.env_name = "CartPole-v1"  # 环境名字
        self.new_step_api = False  # 是否用gym的新api
        self.algo_name = "PPO"  # 算法名字
        self.mode = "train"  # train or test
        self.seed = 1  # 随机种子
        self.train_eps = 200  # 训练的回合数
        self.test_eps = 20  # 测试的回合数
        self.max_steps = 200  # 每个回合的最大步数
        self.eval_eps = 5  # 评估的回合数
        self.eval_per_episode = 10  # 评估的频率

        self.gamma = 0.99  # 折扣因子
        self.k_epochs = 4  # 更新策略网络的次数
        self.actor_lr = 0.0003  # actor网络的学习率
        self.critic_lr = 0.0003  # critic网络的学习率
        self.eps_clip = 0.2  # epsilon-clip
        self.entropy_coef = 0.01  # entropy的系数
        self.update_freq = 100  # 更新频率
        self.actor_hidden_dim = 256  # actor网络的隐藏层维度
        self.critic_hidden_dim = 256  # critic网络的隐藏层维度
        if torch.cuda.is_available():  # 是否使用GPUs
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # self.device = torch.device('cpu')