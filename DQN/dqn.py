import random
import gym
import numpy as np
import collections

from torch.distributions import Categorical
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from RLUtil import rl_utils
import seaborn as sns
from PlotUtil import plot_util


class ReplayBuffer:
    """
    经验回放池
    """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):
        """
        添加数据到缓存
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        # 从buffer中采样数据,数量为batch_size
        :param batch_size:
        :return:
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """
    只有一层隐藏层的Q网络
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

class DQN:
    """
    DQN 算法
    """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        # q 网络
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 目标网络 target q net
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):
        """
         epsilon-贪婪策略采取动作
        :param state:
        :return:
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def predict_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.q_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        # 下个状态的最大q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方差误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1


if __name__ == '__main__':
    print('111')
    lr = 2e-3  # 学习率
    num_episodes = 500  # 回合数
    hidden_dim = 128  # 隐藏层数
    gamma = 0.98  # 折损因子
    epsilon = 0.01  # epsilon贪婪
    target_update = 10  # target net更新频率
    buffer_size = 10000  # 缓存大小
    minimal_size = 500  # 最小大小
    batch_size = 64  # 批大小
    # 设备环境 device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # if torch.cuda.is_available():  # 是否使用GPUs
    #     device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # else:
    #     device = torch.device('cpu')
    device = torch.device('cpu')

    env_name = 'CartPole-v0'
    env = gym.make(env_name, render_mode="human")
    random.seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    env.render()
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    env.close()
    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('DQN on {}'.format(env_name))
    # plt.show()
    #
    # mv_return = rl_utils.moving_average(return_list, 9)
    # plt.plot(episodes_list, mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('DQN on {}'.format(env_name))
    # plt.show()

    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"training curve on {device} for {env_name}")
    plt.xlabel('epsiodes')
    plt.plot(return_list, label='rewards')
    plt.plot(plot_util.smooth(return_list), label='smoothed')
    plt.legend()
    plt.show()

    # rewards = []  # 记录所有回合的奖励
    # steps = []
    # for i_ep in range(10):
    #     ep_reward = 0  # 记录一回合内的奖励
    #     ep_step = 0
    #     state, _ = env.reset()  # 重置环境，返回初始状态
    #     for _ in range(num_episodes):
    #         ep_step += 1
    #         action = agent.predict_action(state)  # 选择动作
    #         next_state, reward, done, _, _ = env.step(action)  # 更新环境，返回transition
    #         state = next_state  # 更新下一个状态
    #         ep_reward += reward  # 累加奖励
    #         env.render()
    #         if done:
    #             break
    #     steps.append(ep_step)
    #     rewards.append(ep_reward)
    #     print(f"回合：{i_ep + 1}/{10}，奖励：{ep_reward:.2f}")
    # print("完成测试")
    # env.close()
    # sns.set()
    # plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"testing curve on {device} for {env_name}")
    # plt.xlabel('epsiodes')
    # plt.plot(rewards, label='rewards')
    # plt.plot(plot_util.smooth(rewards), label='smoothed')
    # plt.legend()
    # plt.show()