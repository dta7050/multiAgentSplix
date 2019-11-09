"""This block of code implements the asynchronous Q-Learning algorithm
using the threading module for multi-processing. FunctionApproximator.py is used
to acquire the Q values and update the gradients of the target network and the
policy network. Actions are chosen according to Epsilon-Greedy selection
At the end, the trained snake is rendered graphically to visually check for behaviour
"""

import tensorflow as tf
import numpy as np
import random
import os
import shutil

import Agent
import FunctionApproximator
import Game
import Constants

from threading import Lock, Thread, get_ident
from queue import Queue
from multiprocessing import cpu_count
from typing import List

from Action import Action
from Snake import Snake
from FunctionApproximator import NeuralNetwork


def epsilon_greedy_action(snake: Snake, sess: tf.Session, nn: NeuralNetwork, state: np.ndarray, epsilon: List[float]):
    """

    :param snake:
    :param sess:
    :param nn:
    :param state:
    :param epsilon:
    :return:
    """
    state = [state]  # List[np.ndarray]
    possible_actions = snake.permissible_actions()  # type: List[Action]
    best_action, _ = nn.max_permissible_Q(sess, state, possible_actions)
    best_action = Action(best_action)
    # possible_actions.remove(best_action)
    prob = random.uniform(0, 1)
    # choose action according to epsilon-greedy
    # return the action to be chosen
    if prob <= epsilon:
        return best_action
    else:
        return random.choice(possible_actions)


def best_q(snake: Snake, sess, nn, state):
    """

    :param snake:
    :param sess:
    :param nn:
    :param state:
    :return:
    """
    # state = [state]
    return nn.max_permissible_Q(sess, state, snake.permissible_actions())[1]


def async_Q(max_time_steps: int, reward: int, penalty: int, checkpointFrequency: int, checkpoint_dir: str,
            policyNetwork: List[NeuralNetwork], policySess: List[tf.Session], targetNetwork: List[NeuralNetwork],
            targetSess: List[tf.Session], lock: Lock, queue: Queue):
    """

    :param max_time_steps:
    :param reward:
    :param penalty:
    :param checkpointFrequency:
    :param checkpoint_dir:
    :param policyNetwork:
    :param policySess:
    :param targetNetwork:
    :param targetSess:
    :param lock:
    :param queue:
    :return:
    """
    time_steps = 0
    epsilon = []  # type: List[float]
    for idx in range(Constants.numberOfSnakes):
        while True:
            e = np.random.normal(0.8, 0.1)
            if e < 1:
                epsilon.append(e)
                break

    while True:
        g = Game.Game()
        #Start a Game
        snake_list = g.snakes
        episodeRunning = True
        pastStateAlive = [True for i in range(Constants.numberOfSnakes)]
        actions_taken = [0 for j in range(Constants.numberOfSnakes)]
        initial_state = [0]*Constants.numberOfSnakes

        state           = [ [] for _ in range(Constants.numberOfSnakes) ]
        action          = [ [] for _ in range(Constants.numberOfSnakes) ]
        reward          = [ [] for _ in range(Constants.numberOfSnakes) ]
        next_state_Q    = [ [] for _ in range(Constants.numberOfSnakes) ]

        while episodeRunning: #Meaning in an episode
            for idx in range(Constants.numberOfSnakes):
                pruned_snake_list = [ snake for snake in snake_list if snake != snake_list[idx] ]
                if g.snakes[idx].alive:
                    initial_state[idx] = Agent.getState(g.snakes[idx], pruned_snake_list, g.food, normalize=True)
                    actions_taken[idx] = epsilon_greedy_action(g.snakes[idx], policySess[idx], policyNetwork[idx], initial_state[idx], epsilon[idx])

                    state[idx].append(initial_state[idx])
                    action[idx].append([actions_taken[idx]])
                    pastStateAlive[idx] = True
                else:
                    actions_taken[idx] = None
                    pastStateAlive[idx] = False

            try:
                single_step_reward, episodeRunning = g.move(actions_taken)
            except AssertionError:
                print("Error making moves {} in game :\n{}".format(actions_taken, g))

            #Now we transition to the next state
            time_steps += 1
            lock.acquire()
            T = queue.get()
            T += 1
            queue.put(T)
            lock.release()

            if T % checkpointFrequency == 0:
                for idx in range(Constants.numberOfSnakes):
                    policyNetwork[idx].save_model(policySess[idx], "{}/policy_{}_{}.ckpt".format(checkpoint_dir, T, idx))
                    targetNetwork[idx].save_model(targetSess[idx], "{}/target_{}_{}.ckpt".format(checkpoint_dir, T, idx))

            for idx in range(Constants.numberOfSnakes):
                if (pastStateAlive[idx]): # To check if snake was already dead or just died
                    reward[idx].append([single_step_reward[idx]])

                    pruned_snake_list = [ snake for snake in snake_list if snake != snake_list[idx] ]
                    if not episodeRunning or not g.snakes[idx].alive: # train on terminal
                        next_state_Q[idx].append([0])
                        lock.acquire()
                        policyNetwork[idx].train(policySess[idx], state[idx], action[idx], reward[idx], next_state_Q[idx])
                        lock.release()
                        state[idx], action[idx], reward[idx], next_state_Q[idx] = [], [], [], []
                    else:
                        final_state = Agent.getState(g.snakes[idx], pruned_snake_list, g.food, normalize=True)
                        next_state_best_Q = best_q(g.snakes[idx], targetSess[idx], targetNetwork[idx], [final_state])
                        next_state_Q[idx].append([next_state_best_Q])

            if time_steps % Constants.AQ_asyncUpdateFrequency == 0:
                for idx in range(Constants.numberOfSnakes):
                    if pastStateAlive[idx] and g.snakes[idx].alive and episodeRunning: # train only if non-terminal, since terminal case is handled above
                        lock.acquire()
                        policyNetwork[idx].train(policySess[idx], state[idx], action[idx], reward[idx], next_state_Q[idx])
                        lock.release()
                    state[idx], action[idx], reward[idx], next_state_Q[idx] = [], [], [], []

            T = queue.get()
            queue.put(T)
            if T % Constants.AQ_globalUpdateFrequency == 0:
                for idx in range(Constants.numberOfSnakes):
                    checkpoint_path = "{}/transfer_{}.ckpt".format(checkpoint_dir, idx)
                    lock.acquire()
                    policyNetwork[idx].save_model(policySess[idx], checkpoint_path)
                    targetNetwork[idx].restore_model(targetSess[idx], checkpoint_path)
                    lock.release()

            T = queue.get()
            queue.put(T)
            if T >= max_time_steps:
                break

        print("Episode done on thread {}. T = {}.".format(get_ident(), T))
        T = queue.get()
        queue.put(T)
        if T >= max_time_steps:
            break
    print("Thread {} complete.".format(get_ident()))
    for idx in range(Constants.numberOfSnakes):
        policyNetwork[idx].save_model(policySess[idx], "{}/policy_{}_{}.ckpt".format(checkpoint_dir, T+1, idx))
        targetNetwork[idx].save_model(targetSess[idx], "{}/target_{}_{}.ckpt".format(checkpoint_dir, T+1, idx))


def train(max_time_steps: int = 1000, reward: int = 1, penalty: int = -10,
          size_of_hidden_layer: int = 20, num_threads: int = 4, checkpointFrequency: int = 500,
          checkpoint_dir: str = "checkpoints", load: bool = False, load_dir: str = "checkpoints",
          load_time_step: int = 500):
    """

    :param max_time_steps:
    :param reward:
    :param penalty:
    :param size_of_hidden_layer:
    :param num_threads:
    :param checkpointFrequency:
    :param checkpoint_dir:
    :param load:
    :param load_dir:
    :param load_time_step:
    :return:
    """
    policyNetwork = []  # type: List[NeuralNetwork]
    targetNetwork = []  # type: List[NeuralNetwork]
    policySess = []     # type: List[tf.Session]
    targetSess = []     # type: List[tf.Session]

    if os.path.isdir(checkpoint_dir):
        # if directory exists, delete it and its contents
        try:
            shutil.rmtree(checkpoint_dir)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))
    os.makedirs(checkpoint_dir)

    length = Agent.getStateLength()
    #Initializing the 2*n neural nets
    for idx in range(Constants.numberOfSnakes):
        policyNetwork.append(FunctionApproximator.NeuralNetwork(length, size_of_hidden_layer))
        targetNetwork.append(FunctionApproximator.NeuralNetwork(length, size_of_hidden_layer))
        policySess.append(tf.Session(graph=policyNetwork[idx].graph))
        targetSess.append(tf.Session(graph=targetNetwork[idx].graph))
        policyNetwork[idx].init(policySess[idx])
        targetNetwork[idx].init(targetSess[idx])
        checkpoint_path = "{}/transfer_{}.ckpt".format(checkpoint_dir, idx)
        policyNetwork[idx].save_model(policySess[idx], checkpoint_path)
        targetNetwork[idx].restore_model(targetSess[idx], checkpoint_path)

    if load:  # resume training from old checkpoints
        for idx in range(Constants.numberOfSnakes):
            policyNetwork[idx].restore_model(policySess[idx], "{}/policy_{}_{}.ckpt".format(load_dir, load_time_step, idx))
            targetNetwork[idx].restore_model(targetSess[idx], "{}/target_{}_{}.ckpt".format(load_dir, load_time_step, idx))

    T = 0
    q = Queue()
    q.put(T)
    lock = Lock()
    threads = [Thread(target=async_Q, args=(max_time_steps, reward, penalty, checkpointFrequency, checkpoint_dir,
                                            policyNetwork, policySess, targetNetwork, targetSess, lock, q)) for _ in range(num_threads)]
    # map(lambda t: t.start(), threads)
    for t in threads:
        t.start()

    print(threads)
    print("main complete")


def graphical_inference(size_of_hidden_layer: int = 20, load_dir: str = "checkpoints", load_time_step: int = 500,
                        play: bool = False, scalingFactor: int = 9):
    """
    To render graphics of the trained agents
    :param size_of_hidden_layer:
    :param load_dir:
    :param load_time_step:
    :param play:
    :param scalingFactor:
    :return:
    """
    import pygame
    import GraphicsEnv

    numSnakes = Constants.numberOfSnakes  # type: int
    if play:
        numSnakes += 1
    colors = np.random.randint(0, 256, size=[numSnakes, 3])
    if play:  # user interacts with the agents
        colors[0] = (0, 0, 0)  # player's snake is always black
    # Create Game Window
    win = pygame.display.set_mode((scalingFactor * Constants.gridSize, scalingFactor * Constants.gridSize))
    screen = pygame.Surface((Constants.gridSize+1, Constants.gridSize+1))  # Grid Screen
    pygame.display.set_caption("Snake Game")
    crashed = False

    targetNetwork = []  # type: List[NueralNetwork]
    targetSess = []     # type: List[tf.Session]
    if play:
        targetNetwork.append(None)
        targetSess.append(None)
    length = Agent.getStateLength()
    for idx in range(int(play), numSnakes):
        targetNetwork.append(FunctionApproximator.NeuralNetwork(length, size_of_hidden_layer=size_of_hidden_layer))
        targetSess.append(tf.Session(graph=targetNetwork[idx].graph))
        targetNetwork[idx].init(targetSess[idx])
        targetNetwork[idx].restore_model(targetSess[idx], "{}/target_{}_{}.ckpt".format(load_dir, load_time_step, idx - int(play)))

    g = Game.Game(numSnakes)
    episodeRunning = True

    while episodeRunning and not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

        actionList = []
        if play:
            actionList.append( GraphicsEnv.manual_action(g.snakes[0], event) )
        for i in range(int(play), numSnakes):
            snake = g.snakes[i]
            if not snake.alive:
                actionList.append(None)
                continue
            opponentSnakes = [opponent for opponent in g.snakes if opponent != snake]
            state = Agent.getState(snake, opponentSnakes, g.food, normalize=True)
            action, _ = targetNetwork[i].max_permissible_Q(targetSess[i], [state], snake.permissible_actions())
            actionList.append(action)

        singleStepRewards, episodeRunning = g.move(actionList)
        GraphicsEnv.displayGame(g, win, screen, colors)


if __name__ == '__main__':
    # train(max_time_steps=500, reward=1, penalty=-10)
    graphical_inference(False, 3, load_dir="checkpoints", load_time_step=500, play=False, scalingFactor=9)
