import numpy as np
from collections import deque
import random

from criticNetwork import CriticNetwork
from actorNetwork import ActorNetwork
from replayBuffer import ReplayBuffer

# parameters
REPLAY_BUFFER_SIZE = 10000
MINI_BATCH_SIZE = 128
GAMMA = 0.99

class DDPGAgent:
    def __init__(self, env):
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.action_max = env.action_space.high
        self.action_min = env.action_space.low

        self.replay_buffer = ReplayBuffer

        self.critic_net = CriticNetwork(self.num_states, self.num_actions, self.action_max, self.action_min)
        self.actor_net = ActorNetwork(self.num_states, self.num_actions, self.action_max)

    def train(self):
        # Prepare minibatch sampling from replay buffer
        batch = self.replay_buffer.minibatch_sample(MINI_BATCH_SIZE)
        s_batch = np.array([item[0] for item in batch])
        a_batch = np.array([item[1] for item in batch])
        reward_batch = np.array([[item[2]] for item in batch])
        s_1_batch = np.array([item[3] for item in batch])
        done_batch = np.array([item[4] for item in batch])

        # Train learned critic network
        q_t_1 = self.critic_net.forward_target_net(s_1_batch, self.actor_net.forward_target_net(s_1_batch))
        target_q_batch = []
        for i in range(MINI_BATCH_SIZE):
            if done_batch[i]:
                target_q_batch.append(reward_batch[i][0])
            else:
                target_q_batch.append(reward_batch[i][0] + GAMMA * q_t_1[i][0])
        target_q_batch = np.reshape(target_q_batch, [MINI_BATCH_SIZE, 1])
        self.critic_net.train(s_batch, a_batch, target_q_batch)

        # Train learned actor network (Deterministic Policy Gradient theorem is applied here)
        dQ_da_batch = self.critic_net.compute_dQ_da(s_batch, self.actor_net.forward_learned_net(s_batch))
        self.actor_net.train(s_batch, dQ_da_batch)

        # Update target networks making closer to learned networks
        self.critic_net.update_target_net()
        self.actor_net.update_target_net()