from collections import deque, namedtuple
import random
import numpy as np


class Memory:
    buffer = deque(maxlen=10000)
    Sample = namedtuple('Sample', ['current_obs', 'current_action', 'next_obs', 'reward', 'done'])

    def add(self, curr_obs, curr_action, next_obs, reward, done=False):
        if self.buffer.full():
            self.buffer.popleft()

        self.buffer.append(self.Sample(curr_obs, curr_action, next_obs, reward, done))

    def sample(self, batch_size=32):
        rand_samp = random.sample(self.buffer, batch_size)

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

        current_obs = np.asarray(current_obs)
        current_action = np.asarray(current_action)
        next_obs = np.asarray(next_obs)
        reward = np.asarray(reward)
        done = np.asarray(done)

        return current_obs, current_action, next_obs, reward, done
