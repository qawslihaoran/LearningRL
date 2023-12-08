import random
from collections import deque


class ReplayBufferQue:
    """
    DQN的经验回放池, 每次采样batch_size个样本
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions):
        self.buffer.append(transitions)

    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential:  # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class PGReplay(ReplayBufferQue):
    """
    Poilcy Gradient经验回放池, 每次采样所有样本
    """

    def __init__(self):
        self.buffer = deque()

    def sample(self, **kwargs):
        batch = list(self.buffer)
        return zip(*batch)
