import torch
from torch.distributions import Bernoulli
from torch.autograd import Variable
import numpy as np


class PolicyGradient:
    def __init__(self, model, memory, cfg):
        self.gamma = cfg['gamma']
        self.device = torch.device(cfg['device'])
        self.memory = memory
        self.policy_net = model.to(self.device)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=cfg['lr'])

    def sample_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state)
        m = Bernoulli(probs)
        action = m.sample()
        action = action.data.numpy().astype(int)[0]  # 转为标量
        return action

    def predict_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state)
        m = Bernoulli(probs)  # 伯努利分布
        action = m.sample()
        action = action.data.numpy().astype(int)[0]  # 转为标量
        return action

    def update(self):
        state_pool, action_pool, reward_pool = self.memory.sample()
        state_pool, action_pool, reward_pool = list(state_pool), list(action_pool), list(reward_pool)
        # discount reward
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma * reward_pool[i]
                reward_pool[i] = running_add

        # normalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # gradient descent
        self.optimizer.zero_grad()  # reset
        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward  # Negative score function x reward
            loss.backward()
        self.optimizer.step()
        self.memory.clear()