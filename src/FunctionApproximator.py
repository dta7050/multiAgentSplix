"""This code is used to initialize and create neural networks
that is used in the Asynchronous Q-Learning algorithm.
It also provides helper methods to accumulate gradients and update them"""

import tensorflow as tf
import numpy as np

import Constants

class NeuralNetwork:

    def __init__(self,data_length, size_of_hidden_layer=20):
        self.graph = tf.Graph()
        with self.graph.as_default():
            layers = self.create_model(data_length, size_of_hidden_layer) # defines the neural network architecture
            action = tf.placeholder(tf.int32, shape=[None,1], name="action_selected")
            Q_value = tf.batch_gather(layers[-1], action, name="Q") # fetch the Q(s,a) value

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=Constants.AQ_lr)

            reward = tf.placeholder(tf.float32, shape=[None, 1], name="reward")
            best_Q = tf.placeholder(tf.float32, shape=[None, 1], name="best_next_state_Q")
            t1 = Constants.gamma * best_Q
            t2 = reward + t1
            difference = t2 - Q_value
            loss = tf.square(difference, name="loss")

            trainable_vars = tf.trainable_variables()
            saver = tf.train.Saver(trainable_vars, max_to_keep=None) # us ed for saving and restoring the weights of the hidden layers
            train_op = optimizer.minimize(loss)

            global_init = tf.global_variables_initializer()
        # Dictionary to access all these layers for running in session
        self.model = {}
        # all placeholders
        self.model["state"] = layers[0]
        self.model["action"] = action
        self.model["reward"] = reward
        self.model["best_Q"] = best_Q
        # all outputs
        self.model["softmax"] = layers[-1]
        self.model["Q_value"] = Q_value
        self.model["train"] = train_op
        # saver
        self.model["saver"] = saver
        # initializer
        self.model["init"] = global_init
        return

    def create_model(self, data_length, size_of_hidden_layer):
        layers = [0, 0, 0, 0]
        xavier_init = tf.contrib.layers.xavier_initializer()
        layers[0] = tf.placeholder(tf.float32, shape=[None,data_length], name="data")
        layers[1] = tf.layers.dense(layers[0], size_of_hidden_layer,
                                                            kernel_initializer=xavier_init, use_bias=True,
                                                            activation=tf.nn.relu, name="hidden")
        layers[2] = tf.layers.dense(layers[1], 4, # one for each action
                                                            kernel_initializer=xavier_init, use_bias=True,
                                                            activation=tf.nn.relu, name="out")
        layers[3] = tf.nn.softmax(layers[2], name="softmax")
        return layers

    def save_model(self, sess, path):
        save_path = self.model["saver"].save(sess, path)
        return save_path

    def restore_model(self, sess, path):
        self.model["saver"].restore(sess, path)
        return

    def init(self, sess):
        sess.run(self.model["init"])
        return

    def Q(self, sess, state, action):
        return sess.run(self.model["Q_value"], feed_dict={ self.model["state"] : state, self.model["action"] : action })

    def max_permissible_Q(self, sess, state, permissible_actions): # NOTE: implemented only for batch_size=1
        Q_values = sess.run(self.model["softmax"], feed_dict={ self.model["state"] : state })[0]
        permissible_actions = list(map(int, permissible_actions))
        permissible_Q = Q_values[permissible_actions]
        best_Q_index = np.argmax(permissible_Q)
        best_action = permissible_actions[best_Q_index]
        return (best_action, permissible_Q[best_Q_index])

    def train(self, sess, state, action, reward, next_state_Q):
        arg_dict = {self.model["state"] : state,
                            self.model["action"] : action,
                            self.model["reward"] : reward,
                            self.model["best_Q"] : next_state_Q}
        return sess.run(self.model["train"], feed_dict=arg_dict)
