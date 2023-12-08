import matplotlib.pyplot as plt
import seaborn as sns


def smooth(data, weight=0.9):
    """
    用于平滑曲线，类似于Tensorboard中的smooth曲线
    :param data:
    :param weight:
    :return:
    """
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards(rewards, cfg, tag='train'):
    """
    画图
    :param rewards:
    :param cfg:
    :param tag:
    :return:
    """
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()
