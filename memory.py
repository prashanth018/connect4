from collections import deque, namedtuple
import random
import numpy as np


class Memory:

    def __init__(self, batch_size=32):
        self.buffer = deque(maxlen=100000)
        self.batch_size = batch_size
        self.Sample = namedtuple('Sample', ['current_obs', 'current_action', 'next_obs', 'reward', 'done'])

    def add(self, curr_obs, curr_action, next_obs, reward, done=False):
        if len(self.buffer) == 10000:
            self.buffer.popleft()

        self.buffer.append(self.Sample(curr_obs, curr_action, next_obs, reward, done))

    def sample(self):
        rand_samp = random.sample(list(self.buffer), self.batch_size)

        current_obs = []
        current_action = []
        next_obs = []
        reward = []
        done = []

        for i in range(32):
            current_obs.append(rand_samp[i].current_obs)
            current_action.append(rand_samp[i].current_action)
            next_obs.append(rand_samp[i].next_obs)
            reward.append(rand_samp[i].reward)
            done.append(rand_samp[i].done)

        current_obs = np.asarray(current_obs, dtype=np.float32)
        current_action = np.asarray(current_action, dtype=np.int32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done)

        return current_obs, current_action, next_obs, reward, done


if __name__ == "__main__":
    test = Memory()
    for i in range(100):
        curr_obs = np.random.rand(6, 7)
        curr_state = random.randint(-1, 2)
        next_obs = np.random.rand(6, 7)
        reward = random.randint(0, 4)
        done = True
        test.add(curr_obs, curr_state, next_obs, reward, done)

    # print(test.sample())
    # co, cs, no, re, do = test.sample()
