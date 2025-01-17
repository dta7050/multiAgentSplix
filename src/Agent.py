''' This file contains methods to compute the state space for a given
snake. 'getState' method is called with the arguments that specify
if its a multiagent setting, if relative or absolute state space has to
be used, if normalisation has to be applied, along with the other arguments.
The description of the state space for the various cases considered are
as below: '''

###   SINGLE SNAKE  ###
    # absolute
        # head
        # k nearest points - from head
        # direction - action enums
    # relative
        # k nearest points
        # direction - action enums
        # nearest wall

###  MULITPLE SNAKES  ###
    # absolute
        # head
        # k nearest points
        # direction - action enums
        # other snakes head
        # other snakes closest point
        # direction of the other agent
        # nearest point of the snake to the other agent
    # relative
        # k nearest points
        # direction - action enums
        # other snakes head
        # other snakes closest point
        # direction of the other agent
        # nearest point of the snake to the other agent
        # nearest wall

from math import *
import numpy as np

import Constants

from Snake import Snake
from Point import Point
from Action import Action
from Food import Food
from numpy import ndarray

from typing import List

''' Given two points, this method calculates the distance between them '''


def calculateDistance(p1: Point, p2: Point) -> float:
    """
    Calculates the distance between two points
    :param p1: x and y coordinates of first point
    :param p2: x and y coordinates of second point
    :return: The distance between the two points
    """
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


'''This method computes the 'k' nearest food points
to the head of the snake '''


def findKNearestPoints(head: Point, food: Food):
    """
    Find the location of the closest food points
    :param head: Point corresponding to the head of the snake
    :param food: Variable storing information about food in the game
    :return: List of the closest food points
    """
    dist = []
    nearestPoints = []
    k = Constants.numNearestFoodPointsForState  # the number of food points to keep track of

    for f in food.foodList:
        dist.append(calculateDistance(f, head))  # calculates the distance between the food and snake head

    if len(food.foodList) < k:
        k = len(food.foodList)  # adjusts constant, k, if number of total food points is less than k

    argmin = np.argpartition(dist, k)  # sorts arguments of dist from smallest to largest

    for i in range(k):
        nearestPoints.append(food.foodList[argmin[i]])  # adds the closest food points to the list

    return nearestPoints


''' This method returns the direction of motion of the snake, given the
snake object '''


def findSnakeDirection(snake: Snake) -> int:
    """
    Finds the direction that the snake is moving
    :param snake: The snake in question
    :return: The direction that the snake is moving
    """
    """
    Joint  ->  ------------ <- Head
              |
    ----------|  <- Joint
    """
    if snake.joints == []:  # if the snake has a joint, use that to determine direction
        direction = snake.findDirection(snake.head, snake.end)
    else:
        direction = snake.findDirection(snake.head, snake.joints[0])

    return direction


''' This method is used in case of relative state representation. That is
when the points are represented relative to the head of a snake. This
method returns the relative position of one point with respect to the other '''


def relativePoints(head: Point, point: Point) -> Point:
    """
    Takes a point in the environment and returns
    it's coordinates relative to the snake's head
    :param head: The head of the snake in question
    :param point: The point to be converted
    :return: The coordinates of the point relative
             to the snake's head
    """
    return Point(point.x - head.x, point.y - head.y)


''' Given a set of points, this method returns the point that is closest to
the snake's head. In case of the presence of multiple nearest points, it
returns a point in the direction of the snake's movement '''


def calculateMinDistPoint(snake: Snake, points: List[Point]) -> Point:
    """
    Returns the point closest to the snake's head given a group of points
    :param snake: The snake in question
    :param points: The list of points to be analyzed
    :return: The point closest to the snake's head
    """
    dist = []
    for point in points:
        dist.append(calculateDistance(point, snake.head))  # calculates the distance between each point and the snake
    dist = np.asarray(dist)  # makes the list into a numpy array

    minIndices = np.where(dist == dist.min())  # stores all closest points in an array
    if minIndices[0].shape[0] == 1:  # if there is only one closest point
        return points[minIndices[0][0]]  # return that point
    else:  # if there are multiple closest points
        direction = findSnakeDirection(snake)  # find the direction of the snake
        for index in range(len(minIndices[0])):  # check to see if any point is in the same direction as snake
            if direction == Action.TOP:
                if(points[minIndices[0][index]].x == snake.head.x and points[minIndices[0][index]].y >= snake.head.y):
                    return points[minIndices[0][index]]
            elif direction == Action.DOWN:
                if(points[minIndices[0][index]].x == snake.head.x and points[minIndices[0][index]].y <= snake.head.y):
                    return points[minIndices[0][index]]
            elif direction == Action.RIGHT:
                if(points[minIndices[0][index]].y == snake.head.y and points[minIndices[0][index]].x >= snake.head.x):
                    return points[minIndices[0][index]]
            elif direction == Action.LEFT:
                if(points[minIndices[0][index]].y == snake.head.y and points[minIndices[0][index]].x <= snake.head.x):
                    return points[minIndices[0][index]]

        return points[minIndices[0][0]]  # if not, return the first closest point


''' This method returns the nearest wall point to the snake's head. This
is used in the case of relative representation of points in the state space '''


def findNearestWall(snake: Snake) -> Point:   # checks the perpendicular distance from the
    """
    Finds the outer most points in each direction (up, down, left, right)
    starting from the snake's head and determines which of these is the
    closest
    :param snake: The snake in question
    :return: The location of the closest wall point
    """
    points = []  # empty list to store points
    points.append(Point(0, snake.head.y))  # adds wall point directly to the left of snake's head
    points.append(Point(snake.head.x, 0))  # adds wall point directly below snake's head
    points.append(Point(Constants.gridSize, snake.head.y))  # adds wall point directly to the right of snake's head
    points.append(Point(snake.head.x, Constants.gridSize))  # adds wall point directly above snake's head

    minDistPoint = calculateMinDistPoint(snake, points)  # finds the minimum distance of the points above
    return minDistPoint  # returns the closest point


''' This method returns the nearest body point of the other snakes to the
snake's head. '''


def findOtherSnakeNearestPoint(snake1: Snake, snake2: Snake) -> Point:  # snake2's nearest body point to the head of snake1
    """
    Takes in two snakes and computes the point on the second
    snake that is closest to the first snake's head(?) (doesn't
    explicitly say head)
    :param snake1: The snake that wants to know how close the other snake is
    :param snake2: The snake that the first snake would like to know its distance to
    :return: The closest point on  the second snake to the first snake
    """
    body = [snake2.head]  # creates a representation of the second snake's body
    body.extend(snake2.joints)
    body.append(snake2.end)
    points = Point.returnBodyPoints(body)  # converts the body into a set of points

    minDistPoint = calculateMinDistPoint(snake1, points)  # finds the point on the second snake is closest to the first
    return minDistPoint  # returns that point


''' Returns absolute state representation for a single snake game '''


def getAbsoluteStateForSingleAgent(snake: Snake, food: Food):
    """
    Gets the state of the environment for just one agent snake. State contains
    the point of the snake's head, the nearest points of food, and the snake's
    direction of motion. The "Absolute" state refers to the points in the state
    contain absolute coordinates as opposed to coordinates relative to the snake's
    head
    :param snake: The single snake agent
    :param food: The food points in the environment
    :return: The state of the environment
    """
    state = []
    state.append(snake.head)  # adds snake's head to the list

    if(len(food.foodList)):          # k nearest points
        state.extend(findKNearestPoints(snake.head, food))  # adds the nearest food points to the list

    state.append(findSnakeDirection(snake))   # adds the direction of motion to the list

    return state


''' Returns relative state representation for a single snake game '''


def getRelativeStateForSingleAgent(snake: Snake, food: Food):
    """
    Gets the state of the environment for just one agent snake. State contains
    the nearest points of food, and the snake's direction of motion, and the
    point of the nearest wall. The "Relative" state refers to the points in the
    state contain coordinates relative to the snake's head as opposed to their
    absolute locations in the environment
    :param snake: The single snake agent
    :param food: The points of food in the environment
    :return: The state of the environment
    """
    state = []  # empty list for the state features

    if(len(food.foodList)):          # k nearest points
        relativeFoodPoints = []  # empty list for relative points of each point of food
        absoluteFoodPoints = findKNearestPoints(snake.head, food)  # finds the nearest points of food
        for point in absoluteFoodPoints:
            relativeFoodPoints.append(relativePoints(snake.head, point))  # converts to relative points
        state.extend(relativeFoodPoints)  # adds points to list

    state.append(findSnakeDirection(snake))   # adds direction of snake to the list

    state.append(relativePoints(snake.head, findNearestWall(snake)))  # adds nearest wall point to the list

    return state


''' Returns absolute state representation for a multi snake game '''


def getAbsoluteStateForMultipleAgents(snake: Snake, agentList: List[Snake], food: Food):
    """
    Gets the state of the environment for one snake in a game containing multiple
    agents. State contains the point of the snake's head, the nearest points of
    food, and the snake's direction of motion. The "Absolute" state refers to the
    points in the state contain absolute coordinates as opposed to coordinates
    relative to the snake's head
    :param snake: The agent for which the state of the environment is found
    :param agentList: A list of all the other agents in the environment
    :param food: All of the points of food in the environment
    :return: The state of the environment from the point of view of the current snake
    """
    state = []
    state.append(snake.head)  # adds the head of the current snake to the list

    if(len(food.foodList)):
        state.extend(findKNearestPoints(snake.head, food))  # adds the food points to the list

    state.append(findSnakeDirection(snake))   # adds the snake's direction of motion to the list

    for agent in agentList:
        if agent.alive == True:
            state.append(agent.head)    # other agent's head
            state.append(findOtherSnakeNearestPoint(snake, agent)) # other agent's nearest body point
            state.append(findSnakeDirection(agent))   # direction of the other agent
            state.append(findOtherSnakeNearestPoint(agent, snake))  # nearest body point of the snake to the other agent's head
        else:
            state.extend([Point(-1, -1), Point(-1, -1), -1, Point(-1, -1)])  # negative points indicates dead snake

    return state


''' Returns relative state representation for a multi snake game '''


def getRelativeStateForMultipleAgents(snake: Snake, agentList: List[Snake], food: Food):
    """
    Gets the state of the environment for one snake in a game containing multiple
    snakes. State contains the nearest points of food, and the snake's direction
    of motion, and the point of the nearest wall. The "Relative" state refers to
    the points in the state contain coordinates relative to the snake's head as
    opposed to their absolute locations in the environment
    :param snake: The agent for which the state of the environment is found
    :param agentList: A list of all the other agents in the environment
    :param food: All of the points of food in the environment
    :return: The state of the environment from the point of view of the current snake
    """
    state = []

    if(len(food.foodList)):  # finds the nearest food points
        relativeFoodPoints = []
        absoluteFoodPoints = findKNearestPoints(snake.head, food)
        for point in absoluteFoodPoints:
            relativeFoodPoints.append(relativePoints(snake.head, point))  # converts location to relative coordinates
        state.extend(relativeFoodPoints)  # add points to the list

    state.append(findSnakeDirection(snake))  # adds direction of snake to the list

    for agent in agentList:
        if agent.alive == True:
            state.append(relativePoints(snake.head, agent.head))    # other agent's head
            state.append(relativePoints(snake.head,findOtherSnakeNearestPoint(snake, agent))) # other agent's nearest  body point
            state.append(findSnakeDirection(agent))   # direction of the other agent
            state.append(relativePoints(snake.head,findOtherSnakeNearestPoint(agent, snake)))  # nearest body point of the snake to the other agent's head
        else:
            state.extend([Point(-1, -1), Point(-1, -1), -1, Point(-1, -1)])  # negative points indicates dead snake

    state.append(relativePoints(snake.head,findNearestWall(snake)))  # adds nearest wall point to the list

    return state


''' Returns the length of the state according to if the game is single or
multiple snake game '''


def getStateLength() -> int:
    """
    Returns the number of features in the current state
    :return: the number of features in the current state
    """
    if Constants.existsMultipleAgents == False:
        return 9
    elif Constants.existsMultipleAgents == True:
        return 3 + (Constants.numNearestFoodPointsForState*2) + (Constants.numberOfSnakes-1)*7


''' This method is called with the arguments that specify if its a
multiagent setting, if relative or absolute state space has to be used,
if normalisation has to be applied, along with the other arguments'''


def getState(snake: Snake, agentList: List[Snake], food: Food, normalize: bool = False) -> ndarray:
    """
    Gets the state for the snake in question and if the normalize parameter is True,
    the points and the actions in the state are normalized to be between zero and one
    :param snake: The agent for which the state of the environment is found
    :param agentList: List containing the other agent snakes
    :param food: The food in the environment
    :param normalize: Whether or not the state parameters should be normalized
    :return: The state as an ndarray
    """
    state = []

    if snake.alive == False:  # checks if the snake is alive
        return [-1] * getStateLength()
    # checks for multiple agents and whether or not the relative state is to be used
    if Constants.useRelativeState == False and Constants.existsMultipleAgents == False:
        state.extend(getAbsoluteStateForSingleAgent(snake, food))
    elif Constants.useRelativeState == False and Constants.existsMultipleAgents == True:
        state.extend(getAbsoluteStateForMultipleAgents(snake, agentList, food))
    elif Constants.useRelativeState == True and Constants.existsMultipleAgents == False:
        state.extend(getRelativeStateForSingleAgent(snake, food))
    elif Constants.useRelativeState == True and Constants.existsMultipleAgents == True:
        state.extend(getRelativeStateForMultipleAgents(snake, agentList, food))

    flatState = []
    for entry in state:  # normalize the data if desired
        if isinstance(entry, Point):
            if normalize:
                flatState.append(entry.x * 1.0 / Constants.gridSize)
                flatState.append(entry.y * 1.0 / Constants.gridSize)
            else:
                flatState.append(entry.x)
                flatState.append(entry.y)
        elif isinstance(entry, Action):
            if normalize:
                flatState.append(entry.value * 1.0 / 3)
            else:
                flatState.append(entry.value)
        else:
            flatState.append(entry)

    return np.asarray(flatState)
