from collections import namedtuple
import random
import glob
import io
import base64




class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.memory = []
        self.position = 0

    def push(self, *args):
        # Saves a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        # Update the pointer to the next position in the replay memory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)