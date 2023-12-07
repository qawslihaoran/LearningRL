import gym
import GameEnvs.env_cliff_walking_wapper
import sarsa
import sarsa_config
import sarsa_train


def env_agent_config(cfg, seed=1):
    """
    创建环境和智能体
    :param cfg: [description]
    :param seed: 随机种子. Defaults to 1
    :return: env [type]: 环境, agent : 智能体
    """
    game_env = gym.make(cfg.env_name)
    game_env = GameEnvs.env_cliff_walking_wapper.CliffWalkingWapper(game_env)
    n_states = game_env.observation_space.n  # 状态维度
    n_actions = game_env.action_space.n  # 动作维度
    print(f"状态数：{n_states}，动作数：{n_actions}")
    agent = sarsa.Sarsa(n_actions, cfg)
    return game_env, agent


if __name__ == '__main__':
    # 获取参数
    cfg = sarsa_config.get_args()
    # 训练
    env, agent = env_agent_config(cfg)
    res_dic = sarsa_train.train(cfg, env, agent)
    sarsa_config.plot_rewards(res_dic['rewards'], cfg, tag="train")

    # 测试
    res_dic = sarsa_train.test(cfg, env, agent, is_render=True)
    sarsa_config.plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果
