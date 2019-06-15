import queue
import collections


class Memory:
    buffer = queue.Queue(maxsize=10000)
    Sample = collections.namedtuple('Sample', ['current_state', 'current_action', 'next_state', 'reward', 'done'])

    def add(self, curr_state, curr_action, next_state, reward, done=False):
        self.Sample()
        if self.buffer.full():
            self.buffer.get()
        self.buffer.put(curr_state, curr_action, next_state, reward, done)
