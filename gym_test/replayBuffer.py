import numpy as np

''' experience replay buffer '''

class ReplayBuffer(object):
    # init replay buffer - memory
    def __init__(self, buffer_size, buffer_dims):
        self.buffer_size = buffer_size
        self.memory = np.zeros((buffer_size, buffer_dims))
        self.pointer = 0

    # replace the old memory with new memory
    # store transition in memory set
    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, done))
        index = self.pointer % self.buffer_size
        self.memory[index, :] = transition
        self.pointer += 1

    # sample a random minibatch of minibatch_size transitions from memory
    def minibatch_sample(self, minibatch_size):
        assert self.pointer >= self.buffer_size, 'Memory (Experience replay buffer) is not full!'
        random_index = np.random.choice(self.buffer_size, size = minibatch_size)
        return self.memory[random_index, :]