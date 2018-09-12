import numpy as np
import tensorflow as tf

# paramters
HIDDEN_LAYER_1 = 400
HIDDEN_LAYER_2 = 300
MINI_BATCH_SIZE = 64
LEARNING_RATE = 1e-4
TAU = 0.001
EPSILON = 1e-6

class ActorNetwork:
    def __init__(self, sess, s_dim, a_dim, a_high):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_high = a_high

        self.graph = tf.Graph()

        # Learned network
        self.W1, self.B1, self.W2, self.B2, self.W3, self.B3, self.actor_model, self.state_input = self.__create_graph()

        # Target network
        self.t_W1, self.t_B1, self.t_W2, self.t_B2, self.t_W3, self.t_B3, self.t_actor_model, self.t_state_input = self.__create_graph()

        # Update accroding to DDPG theorem
        self.dQ_da_input = tf.placeholder(tf.float32, [None, self.a_dim])
        params = [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3]
        param_grads_sum_over_batch = tf.gradients(self.actor_model, params, -self.dQ_da_input)
        param_grads = [tf.divide(param_grad, MINI_BATCH_SIZE) for param_grad in param_grads_sum_over_batch]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).apply_gradients(zip(param_grads, params))

        # Update target network making closer to learned network
        self.target_net_update_ops = [
            self.t_W1.assign(TAU * self.W1 + (1-TAU) * self.t_W1),
            self.t_W2.assign(TAU * self.W2 + (1-TAU) * self.t_W2),
            self.t_W3.assign(TAU * self.W3 + (1-TAU) * self.t_W3),
            self.t_B1.assign(TAU * self.B1 + (1-TAU) * self.t_B1),
            self.t_B2.assign(TAU * self.B2 + (1-TAU) * self.t_B2),
            self.t_B3.assign(TAU * self.B3 + (1-TAU) * self.t_B3),
        ]

        # Global variables have to be initialized
        self.sess.run(tf.global_variables_initializer())

        # Confirm learned/target net have the same values
        self.sess.run([
            self.t_W1.assign(self.W1),
            self.t_W2.assign(self.W2),
            self.t_W3.assign(self.W3),
            self.t_B1.assign(self.B1),
            self.t_B2.assign(self.B2),
            self.t_B3.assign(self.B3),
        ])

    def __create_graph(self):
        state_input = tf.placeholder(tf.float32, [None, self.s_dim])

        W1 = tf.Variable(tf.truncated_normal([self.s_dim, HIDDEN_LAYER_1], stddev=0.01))
        B1 = tf.Variable(tf.constant(0.03, shape=[HIDDEN_LAYER_1]))
        W2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_1, HIDDEN_LAYER_2], stddev=0.01))
        B2 = tf.Variable(tf.constant(0.03, shape=[HIDDEN_LAYER_2]))
        W3 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_2, self.a_dim], stddev=0.01))
        B3 = tf.Variable(tf.constant(0.03, shape=[self.a_dim]))

        z1 = tf.nn.relu(tf.matmul(state_input, W1) + B1)
        z2 = tf.nn.relu(tf.matmul(z1, W2) + B2)
        actor_model = tf.multiply(self.a_high, tf.nn.tanh(tf.matmul(z2, W3) + B3))

        return W1, B1, W2, B2, W3, B3, actor_model, state_input

    def forward_target_net(self, state_batch):
        return self.sess.run(self.t_actor_model, feed_dict={self.t_state_input: state_batch})

    def forward_learned_net(self, state_batch):
        return self.sess.run(self.actor_model, feed_dict={self.state_input: state_batch})

    def train(self, state_batch, dQ_da_batch):
        self.sess.run(self.optimizer, feed_dict={self.state_input: state_batch, self.dQ_da_input: dQ_da_batch})

    def update_target_net(self):
        self.sess.run(self.target_net_update_ops)