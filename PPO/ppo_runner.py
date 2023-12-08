import gym
import os
import numpy as np
import torch
import random
import ppo_agent
import ppo_config
import ppo_train
from PlotUtil.plot_util import plot_rewards



def all_seed(env, seed=1):
    """
    万能的seed函数
    :param env:
    :param seed:
    :return:
    """
    if seed == 0:
        return
    # env.seed(seed)  # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # config for CPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts


def env_agent_config(cfg):
    env = gym.make(cfg.env_name)  # 创建环境
    all_seed(env, seed=cfg.seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")
    # 更新n_states和n_actions到cfg参数中
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions)
    agent = ppo_agent.Agent(cfg)
    return env, agent


if __name__ == '__main__':
    # 参数
    cfg = ppo_config.Config()
    # 训练
    env, agent = env_agent_config(cfg)
    best_agent, res_dic = ppo_train.train(cfg, env, agent)
    plot_rewards(res_dic['rewards'], cfg, tag="train")

    # 测试
    res_dic = ppo_train.test(cfg, env, best_agent, is_render=False)
    plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果

