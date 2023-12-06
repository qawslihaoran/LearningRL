import QLearning.config
import env_cliff_walking_wapper
import q_learning


def train(cfg, env, agent, is_render=False):
    print('开始训练!')
    print(f'环境{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')
    rewards = []  # 记录奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个回合的奖励 (累积奖励)
        state = env.reset(seed=cfg.seed)  # 重置环境,即开始新的回合
        while True:
            # 根据算法采样一个动作
            action = agent.sample_action(state)
            # 与环境进行一次动作交互
            next_state, reward, terminated, _, _ = env.step(action)
            # Q-Learning算法更新
            agent.learn(state, action, reward, next_state, terminated)
            # 更新状态
            state = next_state
            ep_reward += reward
            if is_render:
                env.render()
            if terminated:
                break
        rewards.append(ep_reward)
        if (i_ep + 1) % 20 == 0:
            print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.1f}，Epsilon：{agent.epsilon:.3f}")
    print('完成训练！')
    return {"rewards": rewards}


def test(cfg, env, agent, is_render=False):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset(seed=cfg.seed)  # 重置环境, 重新开一局（即开始新的一个回合）
        while True:
            action = agent.predict_action(state)  # 根据算法选择一个动作
            next_state, reward, terminated, _, info = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            if is_render:
                env.render()
            if terminated:
                break
        rewards.append(ep_reward)
        print(f"回合数：{i_ep + 1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return {"rewards": rewards}
