"""
This file is used to initialize the neural network that will be used to train our agents.
It is derived from FunctionApproximator.py but is modified to achieve our specific project
goals.
"""

import tensorflow as tf
import numpy as np

import Constants
from typing import List


class NeuralNetwork:

    def __init__(self, num_layers: int = 4, size_of_layers: List[int] = [9, 20, 4, 1], initializer: str = 'xavier', optimizer: str = 'gradient descent'):
        """
        Initializes the network
        :param data_length: Length of the input data to the network
        :param size_of_hidden_layer: Number of neurons in the hidden layer
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            layers = self.create_model(num_layers, size_of_layers, initializer)  # defines the neural net architecture
            action = tf.placeholder(tf.int32, shape=[None, 1], name="action_selected")  # to be defined later
            Q_value = tf.batch_gather(layers[-1], action, name="Q")  # fetch the Q(s,a) value

            if optimizer == 'gradient descent':
                opt = tf.train.GradientDescentOptimizer(learning_rate=Constants.AQ_lr)  # use gradient descent
            elif optimizer == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(learning_rate=Constants.AQ_lr)
            elif optimizer == 'adagrad':
                opt = tf.train.AdagradOptimizer(learning_rate=Constants.AQ_lr)
            elif optimizer == 'adagradda':
                opt = tf.train.AdagradDAOptimizer(learning_rate=Constants.AQ_lr)
            elif optimizer == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=Constants.AQ_lr)
            else:
                print("please use appropriate optimizer")

            reward = tf.placeholder(tf.float32, shape=[None, 1], name="reward")
            best_Q = tf.placeholder(tf.float32, shape=[None, 1], name="best_next_state_Q")
            t1 = Constants.gamma * best_Q
            t2 = reward + t1
            difference = t2 - Q_value  # difference between expected Q and actual Q
            #loss = tf.losses.mean_squared_error(labels=Q_value, predictions=t2)
            loss = tf.square(difference, name="loss")  # Loss function is square of difference

            trainable_vars = tf.trainable_variables()  # list of trainable variables
            saver = tf.train.Saver(trainable_vars, max_to_keep=None)  # used for saving and restoring the weights of the hidden layers
            train_op = opt.minimize(loss)  # network tries to minimize loss

            global_init = tf.global_variables_initializer()  # initializes tensorflow global variables
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

    def create_model(self, num_layers: int, size_of_layers: List[int], initializer: str) -> List:
        """
        Actually creates the neural network
        :param data_length: Length of input data to the network
        :param size_of_hidden_layer: Number of neurons in the hidden layer
        :return: A list of the layers
        """
        layers = []  # initialize list of layers

        for i in range(num_layers):
            layers.append(0)  # add a placeholder zero for every layer in the network

        if initializer == 'xavier':
            weight_init = tf.contrib.layers.xavier_initializer()  # initialize weights using Xavier initialization
        elif initializer == 'he':
            weight_init = tf.contrib.layers.variance_scaling_initializer()  # initialize weights using He initialization
        else:
            print("unknown initializer")

        for j in range(num_layers):
            if not j:  # j == 0
                layers[j] = tf.placeholder(tf.float32, shape=[None, size_of_layers[j]], name="data")  # input layer
            elif j < (num_layers - 1):
                layers[j] = tf.layers.dense(layers[j-1], size_of_layers[j],
                                            kernel_initializer=weight_init, use_bias=True,
                                            activation=tf.nn.relu, name=("hidden" + str(j)))  # hidden layer
            elif j == num_layers - 1:
                layers[j] = tf.nn.softmax(layers[j-1], name="softmax")  # performs softmax to determine action to take
            else:
                print("the thing didn't work (NeuralNet.create_model())")

        return layers

    def save_model(self, sess, path):
        """
        Saves the network (weights and layers and stuff)
        :param sess: A training session to be saved
        :param path: The path to save the training session in
        :return: The path where the model was saved
        """
        save_path = self.model["saver"].save(sess, path)
        return save_path

    def restore_model(self, sess, path):
        """
        Loads a saved model
        :param sess: The training model to be loaded
        :param path: The path containing the session to be loaded
        :return: nothing
        """
        self.model["saver"].restore(sess, path)
        return

    def init(self, sess):
        """
        Initializes the tensorflow global variables
        :param sess: The session to be initialized
        :return: nothing
        """
        sess.run(self.model["init"])
        return

    def Q(self, sess, state, action: 'Action'):
        """
        Gets the Q value obtained from the neural network
        :param sess: The training session to run
        :param state: The current state of the environment
        :param action: The action to take
        :return: The calculated Q value
        """
        return sess.run(self.model["Q_value"], feed_dict={self.model["state"]: state, self.model["action"]: action})

    def max_permissible_Q(self, sess, state, permissible_actions: List['Action']) -> ('Action', int):  # NOTE: implemented only for batch_size=1
        """
        Calculates the Q values associated with each possible action and returns the action
        with the highest Q along with the value of that Q
        :param sess: The training session to be run
        :param state: The current state of the environment
        :param permissible_actions: List of possible actions
        :return: The best action and its Q value
        """
        Q_values = sess.run(self.model["softmax"],
                            feed_dict={self.model["state"]: state})[0]  # run session, get Q values
        permissible_actions = list(map(int, permissible_actions))  # creates list of possible actions
        permissible_Q = Q_values[permissible_actions]  # assigns a Q value to each action
        best_Q_index = np.argmax(permissible_Q)  # finds the argument with the highest Q value from the ones above
        best_action = permissible_actions[best_Q_index]  # action with highest Q is the best action
        return (best_action, permissible_Q[best_Q_index])

    def train(self, sess, state, action: 'Action', reward: int, next_state_Q: int):
        """

        :param sess: Training session to be run
        :param state: Current state of the environment
        :param action: Action to try
        :param reward: Expected reward for the action
        :param next_state_Q: Expected Q value for the next state
        :return: The completed training session
        """
        arg_dict = {self.model["state"]: state,
                    self.model["action"]: action,
                    self.model["reward"]: reward,
                    self.model["best_Q"]: next_state_Q}
        return sess.run(self.model["train"], feed_dict=arg_dict)
