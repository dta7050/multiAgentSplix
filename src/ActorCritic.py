""" This file contains the implementation of the actor critic algorithm. It
contains helper methods to get policy and feature vector that are necessary
for the algorithm. It also contains a method to run the game on a graphical user
interface once the agent has been trained """

import numpy as np
import os
import shutil

import Constants

from Agent import *
from Action import Action
from Point import Point
from Game import Game
from Snake import Snake
from numpy import asarray, ndarray

""" Returns the normalised feature vector which is a combination of the state
points and the action"""


def getFeatureVector(state: ndarray, action: Action) -> ndarray:
    """
    Function takes in a state and an action then iterates through each feature of the state and uses that and the
    value of the action to compute each feature in the feature vector.
    :param state: Current state of the environment
    :param action: A potential action the agent can take
    :return: A vector containing feature values
    """
    featureVector = []  # s*a, s^2*a^2
    actionValue = action.value + 1
    for feature in state:
        # feature is already normalized. Multiplying with actionValue/4 ensures it stays normalized
        featureVector.append(feature * actionValue / 4)
        featureVector.append(feature**2 * actionValue**2 / 16)
    return np.asarray(featureVector)


''' Returns the numerical preferences given the feature vector and the theta parameter.
Numerical preferences is a linear function approximation in theta parameter '''


def getNumericalPreferences(snake: Snake, state: ndarray, theta: ndarray) -> list[ndarray]:
    """
    This function computes a feature vector and then uses a vector of theta values and uses them to
    compute of vector of numerical preferences that correspond to how good each feature value is
    :param snake: The current snake
    :param state: The current state of the environment
    :param theta: Array of parameters to quantify how good each state-action pair is
    :return: List of numerical preferences
    """
    numericalPreferenceList = []
    for action in snake.permissible_actions():
        featureVector = getFeatureVector(state, action)
        numericalPreference = np.dot(theta.T, featureVector)  # dot product of theta transpose and featureVector
        numericalPreferenceList.append(numericalPreference)

    return numericalPreferenceList


''' Returns the softmax policy for the given state, action and theta parameter '''


def getPolicy(snake: Snake, state: ndarray, theta: ndarray, action: Action) -> float:
    """
    Computes the probability to take a given action. The more optimal the action, the
    higher the probability it has
    :param snake: The current snake
    :param state: The current state of the environment
    :param theta: Array of parameters to quantify how good each state-action pair is
    :param action: The action in question which is used to calculate the probability
    :return: A decimal number between 0 and 1 corresponding to the probability of
             the action in question
    """
    featureVector = getFeatureVector(state, action)
    numericalPreference = np.dot(theta.T, featureVector)    # h(s, a, theta)
    numericalPreferenceList = getNumericalPreferences(snake, state, theta)

    # e^h(s, a, theta)/ Sum over b(e^h(s, b, theta))
    return (np.exp(numericalPreference) / np.sum(np.exp(numericalPreferenceList)))


''' Returns the value for a given state which acts as a critic in this algorithm '''


def getValueFunction(state: ndarray, w: ndarray) -> float:
    """
    Function used to get a value to update the rewards and
    policy of each agent
    :param state: Current state of the environment
    :param w: Array of values used to update the policy of
             the agents
    :return: A value calculated through the dot product of
             two vectors. Value is used to update rewards
             and policy of each agent
    """
    if np.all(np.asarray(state) == -1):  # all snakes are dead
        return 0

    featureVector = np.asarray(state)
    return np.dot(w.T, featureVector)


def getAction(snake: Snake, state: ndarray, theta: ndarray) -> Action:
    """
    Function that creates a list of a snakes possible actions, computes the probability of each action,
    and then determines the best action to take given the state of the game
    :param snake: The current snake
    :param state: The current state of the environment
    :param theta: Array of parameters to quantify how good each state-action pair is
    :return: The action to be taken
    """
    actionProbability = []
    actions = []
    for action in snake.permissible_actions():  # iterate through each action
        actionProbability.append(getPolicy(snake, state, theta, action))  # compute the probability of the action
        actions.append(action)  # add the action to the list
    return Action(np.random.choice(actions, p=actionProbability))  # determine which action to take


''' Returns the differential of the policy with respect to the policy parameter theta'''


def getGradientForPolicy(snake: Snake, state: ndarray, action: Action, theta: ndarray) -> ndarray:
    """
    Function that, given an action and a state, computes the gradient which is used to update
    the agent's policy. This is assumed to be analogous to using gradient descent to minimize
    an error function
    :param snake: The current snake
    :param state: The current state of the environment
    :param action: The action for which the calculation will take place
    :param theta: Array of parameters to quantify how good each state-action pair is
    :return: The computed gradient used to update the agent's policy
    """
    featureVector = getFeatureVector(state, action)
    exps = np.exp(getNumericalPreferences(snake, state, theta))
    feature_exps = np.asarray([getFeatureVector(state, action) * exps[i]
                              for i, action in enumerate(snake.permissible_actions())])
    numr = np.sum(feature_exps, axis=0)
    denr = np.sum(exps)
    return featureVector - (numr / denr)


''' Actor critic algorithm is implemented in this method and the agent is set to
train according to the algorithm. It also saves the checkpoints while training '''


def train(maxTimeSteps: int, checkpointFrequency: int = 500, checkpoint_dir="checkpoints",
          load=False, load_dir="checkpoints", load_time_step=500):
    """
    Trains the snakes for a given amount of time. Initializes the training parameters, w and theta,
    to zero, and updates them after every action taken by the snakes
    :param maxTimeSteps: The amount of time the snakes are trained
    :param checkpointFrequency: How often the training parameters are saved
    :param checkpoint_dir: Path of where to save the training data
    :param load: Whether to load pre-trained snakes or train new ones from scratch
    :param load_dir: Path where the training data to load would be
    :param load_time_step: The starting time step upon load
    :return: null
    """
    length = getStateLength()  # returns the length of the state (amount of features the state has)
    theta = np.zeros((Constants.numberOfSnakes, length * 2))  # creates matrix of zeros(num snakes x 2(length of state))
    w = np.zeros((Constants.numberOfSnakes, length))  # creates matrix of zeros (num snakes x length of states)

    if load:  # resume training from old checkpoints
        w = np.load("{}/w_{}.npy".format(load_dir, load_time_step))
        theta = np.load("{}/theta_{}.npy".format(load_dir, load_time_step))

    if os.path.isdir(checkpoint_dir):
        # if directory exists, delete it and its contents
        try:
            shutil.rmtree(checkpoint_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    os.makedirs(checkpoint_dir)

    timeSteps = 0  # initialize the time counter
    counter = 0
    while timeSteps <= maxTimeSteps:
        g = Game()  # initialize the game
        episodeRunning = True

        while episodeRunning:
            I = 1  # used in calculating theta
            actionList = []  # empty list to store actions of each snake
            stateList = []  # empty list to hold each snake's perception of the environment
            for i, snake in enumerate(g.snakes):
                if not snake.alive:
                    actionList.append(None)
                    stateList.append([-1] * getStateLength())
                    continue
                opponentSnakes = [opponent for opponent in g.snakes if opponent != snake]  # finds the opponent snakes
                stateList.append(getState(snake, opponentSnakes, g.food, normalize=True))  # gets the current state
                action = getAction(snake, stateList[i], theta[i])  # compute the snake's action
                actionList.append(action)  # adds the action to the list

            singleStepRewards, episodeRunning = g.move(actionList)  # snakes perform their actions
            timeSteps += 1  # increment the counter
            print("t = " + str(timeSteps))

            if timeSteps % checkpointFrequency == 0:  # save the training parameters
                np.save("{}/theta_{}.npy".format(checkpoint_dir, timeSteps), theta)
                np.save("{}/w_{}.npy".format(checkpoint_dir, timeSteps), w)

            for i, snake in enumerate(g.snakes):  # get the rewards and use them to update theta and w
                if not snake.alive:
                    continue
                opponentSnakes = [opponent for opponent in g.snakes if opponent != snake]
                state = stateList[i]
                action = actionList[i]
                nextState = getState(snake, opponentSnakes, g.food, normalize=True)
                reward = singleStepRewards[i]
                delta = reward + Constants.gamma * getValueFunction(nextState, w[i]) - getValueFunction(state, w[i])
                w[i] = np.add(w[i], (Constants.AC_alphaW * delta) * np.asarray(state))
                theta[i] += Constants.AC_alphaTheta * I * delta * getGradientForPolicy(snake, state, action, theta[i])
                I *= Constants.gamma

            if timeSteps > maxTimeSteps:  # break the loop and end the game
                break

        g.endGame()

    np.save("{}/theta_{}.npy".format(checkpoint_dir, timeSteps), theta)
    np.save("{}/w_{}.npy".format(checkpoint_dir, timeSteps), w)


def inference(load_dir="checkpoints", load_time_step=500):
    w = np.load("{}/w_{}.npy".format(load_dir, load_time_step))
    theta = np.load("{}/theta_{}.npy".format(load_dir, load_time_step))
    g = Game()
    episodeRunning = True
    while episodeRunning:
        actionList = []
        for i, snake in enumerate(g.snakes):
            if not snake.alive:
                actionList.append(None)
                stateList.append([-1] * getStateLength())
                continue
            opponentSnakes = [opponent for opponent in g.snakes if opponent != snake]
            state = getState(snake, opponentSnakes, g.food, normalize=True)
            action = getAction(snake, state, theta[i])
            actionList.append(action)

        singleStepRewards, episodeRunning = g.move(actionList)
        print(g)

''' This method runs the game on a graphical user interface once the agent has been trained'''
def graphical_inference(load_dir="checkpoints", load_time_step=500, play=False, scalingFactor=9):
    import pygame
    import GraphicsEnv

    numSnakes = Constants.numberOfSnakes
    if play:
        numSnakes += 1
    colors = np.random.randint(0, 256, size=[numSnakes, 3])
    if play: # user interacts with the agents
        colors[0] = (0, 0, 0) # player's snake is always black
    win = pygame.display.set_mode((scalingFactor * Constants.gridSize, scalingFactor * Constants.gridSize))  # Game Window
    screen = pygame.Surface((Constants.gridSize+1, Constants.gridSize+1))  # Grid Screen
    pygame.display.set_caption("Snake Game")
    crashed = False

    w = np.load("{}/w_{}.npy".format(load_dir, load_time_step))
    theta = np.load("{}/theta_{}.npy".format(load_dir, load_time_step))
    g = Game(numSnakes)
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
            state = getState(snake, opponentSnakes, g.food, normalize=True)
            action = getAction(snake, state, theta[i - int(play) ])
            actionList.append(action)

        singleStepRewards, episodeRunning = g.move(actionList)
        GraphicsEnv.displayGame(g, win, screen, colors)

if __name__=='__main__':
    graphical_inference(30, False, False, 3, load_dir="old_checkpoints", load_time_step=10000, play=False)
