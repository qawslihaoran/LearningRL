import sarsa


def train(cfg, env, agent: sarsa.Sarsa, is_render=False):
    print('开始训练！')
    print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')
    rewards = []  # 记录奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset()  # 重置环境,即开始新的回合
        action = agent.sample(state)
        while True:
            action = agent.sample(state)  # 根据算法采样一个动作
            next_state, reward, done, _, _ = env.step(action)  # 与环境进行一次动作交互
            next_action = agent.sample(next_state)
            agent.learn(state, action, reward, next_state, next_action, done)  # 算法更新
            state = next_state  # 更新状态
            action = next_action
            ep_reward += reward
            if is_render:
                env.render()
            if done:
                break
        rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.1f}，Epsilon：{agent.epsilon}")
    print('完成训练！')
    return {"rewards": rewards}


def test(cfg, env, agent: sarsa.Sarsa, is_render=False):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个回合）
        while True:
            action = agent.predict(state)  # 根据算法选择一个动作
            next_state, reward, done, _, _ = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            if is_render:
                env.render()
            if done:
                break
        rewards.append(ep_reward)
        print(f"回合数：{i_ep + 1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return {"rewards": rewards}
