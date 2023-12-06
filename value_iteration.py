import numpy as np
import sys
import os

curr_path = os.path.abspath('')
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
from GameEnvs.simple_grid import DrunkenWalkEnv


def all_seed(env, seed=1):
    ## 这个函数主要是为了固定随机种子
    import numpy as np
    import random
    import os
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def value_iteration(env, theta=0.005, discount_factor=0.9):
    Q = np.zeros((env.nS, env.nA))
    count = 0
    while True:
        delta = 0.0
        Q_tmp = np.zeros((env.nS, env.nA))
        for state in range(env.nS):
            for a in range(env.nA):
                accum = 0.0
                reward_total = 0.0
                for prob, next_state, reward, done in env.P[state][a]:
                    accum += prob * np.max(Q[next_state, :])
                    reward_total += prob * reward
                Q_tmp[state, a] = reward_total + discount_factor * accum
                delta = max(delta, abs(Q_tmp[state, a] - Q[state, a]))
        Q = Q_tmp
        count += 1
        if delta < theta or count > 100:
            break
    return Q


def test(env, poilcy, num_episode=1000):
    rewards = []  # 记录所有回合的奖励
    success = []  # 记录该回合是否成功走到终点
    for i_ep in range(num_episode):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个回合） 这里state=0
        while True:
            action = poilcy[state]  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            if done:
                break
        if state == 12:  # 即走到终点
            success.append(1)
        else:
            success.append(0)
        rewards.append(ep_reward)
    acc_suc = np.array(success).sum() / num_episode
    print("测试的成功率是：", acc_suc)


if __name__ == '__main__':
    print('Holle')
    env = DrunkenWalkEnv(map_name="theAlley")
    all_seed(env, seed=1)  # 设置随机种子为1

    Q = value_iteration(env)
    print(Q)

    policy = np.zeros([env.nS, env.nA])
    for state in range(env.nS):
        best_action = np.argmax(Q[state, :])  # 根据价值迭代算法得到的Q表格选择出策略
        policy[state, best_action] = 1

    policy = [int(np.argwhere(policy[i] == 1)) for i in range(env.nS)]
    print(policy)

    test(env, policy)
