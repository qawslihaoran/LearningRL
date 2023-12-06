import config
import gym
import env_cliff_walking_wapper
import q_learning
import train


def env_agent_config(cfg, seed=1):
    game_env = gym.make(cfg.env_name)
    game_env = env_cliff_walking_wapper.CliffWalkingWapper(game_env)
    n_states = game_env.observation_space.n  # 状态维度
    n_actions = game_env.action_space.n  # 动作维度
    agent = q_learning.QLearning(n_states, n_actions, cfg)
    return game_env, agent


if __name__ == '__main__':
    cfg = config.Config()
    # 训练
    env, agent = env_agent_config(cfg)
    res_dic = train.train(cfg, env, agent)
    config.plot_rewards(res_dic['rewards'],
                        title=f"training curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")

    # 测试

    res_dic = train.test(cfg, env, agent, False)
    config.plot_rewards(res_dic['rewards'],
                 title=f"testing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")  # 画出结果