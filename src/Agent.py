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


''' Given two points, this method calculates the distance between them '''
def calculateDistance(p1, p2):
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

'''This method computes the 'k' nearest food points
to the head of the snake '''
def findKNearestPoints(head, food):
    dist = []
    nearestPoints = []
    k = Constants.numNearestFoodPointsForState

    for f in food.foodList:
        dist.append(calculateDistance(f, head))

    if len(food.foodList) < k:
        k = len(food.foodList)

    argmin = np.argpartition(dist, k)

    for i in range(k):
        nearestPoints.append(food.foodList[argmin[i]])

    return nearestPoints

''' This method returns the direction of motion of the snake, given the
snake object '''
def findSnakeDirection(snake):
    if snake.joints == []:
        direction = snake.findDirection(snake.head, snake.end)
    else:
        direction = snake.findDirection(snake.head, snake.joints[0])

    return direction

''' This method is used in case of relative state representation. That is
when the points are represented relative to the head of a snake. This
method returns the relative position of one point with respect to the other '''
def relativePoints(head, point):
    return Point( point.x - head.x, point.y - head.y )

''' Given a set of points, this method returns the point that is closest to
the snake's head. In case of the presence of multiple nearest points, it
returns a point in the direction of the snake's movement '''
def calculateMinDistPoint(snake, points):
    dist = []
    for point in points:
        dist.append(calculateDistance(point, snake.head))
    dist = np.asarray(dist)

    minIndices = np.where(dist == dist.min())
    if minIndices[0].shape[0] == 1:
        return points[minIndices[0][0]]
    else:
        direction = findSnakeDirection(snake)
        for index in range(len(minIndices[0])):
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

        return points[minIndices[0][0]]

''' This method returns the nearest wall point to the snake's head. This
is used in the case of relative representation of points in the state space '''
def findNearestWall(snake):   # checks the perpendicular distance from the
    points = []                             # snake's head to all the walls and returns
    points.append(Point(0, snake.head.y))         # minimum of those
    points.append(Point(snake.head.x, 0))
    points.append(Point(Constants.gridSize, snake.head.y))
    points.append(Point(snake.head.x, Constants.gridSize))

    minDistPoint = calculateMinDistPoint(snake, points)
    return minDistPoint

''' This method returns the nearest body point of the other snakes to the
snake's head. '''
def findOtherSnakeNearestPoint(snake1, snake2): # snake2's nearest body point to the head of snake1
    body = [snake2.head]
    body.extend(snake2.joints)
    body.append(snake2.end)
    points = Point.returnBodyPoints(body)

    minDistPoint = calculateMinDistPoint(snake1, points)
    return minDistPoint

''' Returns absolute state representation for a single snake game '''
def getAbsoluteStateForSingleAgent(snake, food):
    state = []
    state.append(snake.head)    # head

    if(len(food.foodList)):          # k nearest points
        state.extend(findKNearestPoints(snake.head, food))

    state.append(findSnakeDirection(snake))   # direction

    return state

''' Returns relative state representation for a single snake game '''
def getRelativeStateForSingleAgent(snake, food):
    state = []

    if(len(food.foodList)):          # k nearest points
        relativeFoodPoints = []
        absoluteFoodPoints = findKNearestPoints(snake.head, food)
        for point in absoluteFoodPoints:
            relativeFoodPoints.append(relativePoints(snake.head, point))
        state.extend(relativeFoodPoints)

    state.append(findSnakeDirection(snake))   # direction

    state.append(relativePoints(snake.head,findNearestWall(snake)))  # nearest wall point

    return state

''' Returns absolute state representation for a multi snake game '''
def getAbsoluteStateForMultipleAgents(snake, agentList, food):
    state = []
    state.append(snake.head)    # head

    if(len(food.foodList)):          # k nearest points
        state.extend(findKNearestPoints(snake.head, food))

    state.append(findSnakeDirection(snake))   # direction

    for agent in agentList:
        if agent.alive == True:
            state.append(agent.head)    # other agent's head
            state.append(findOtherSnakeNearestPoint(snake, agent)) # other agent's nearest body point
            state.append(findSnakeDirection(agent))   # direction of the other agent
            state.append(findOtherSnakeNearestPoint(agent, snake))  # nearest body point of the snake to the other agent's head
        else:
            state.extend([Point(-1, -1), Point(-1, -1), -1, Point(-1, -1)])

    return state

''' Returns relative state representation for a multi snake game '''
def getRelativeStateForMultipleAgents(snake, agentList, food):
    state = []

    if(len(food.foodList)):          # k nearest points
        relativeFoodPoints = []
        absoluteFoodPoints = findKNearestPoints(snake.head, food)
        for point in absoluteFoodPoints:
            relativeFoodPoints.append(relativePoints(snake.head, point))
        state.extend(relativeFoodPoints)

    state.append(findSnakeDirection(snake))   # direction

    for agent in agentList:
        if agent.alive == True:
            state.append(relativePoints(snake.head, agent.head))    # other agent's head
            state.append(relativePoints(snake.head,findOtherSnakeNearestPoint(snake, agent))) # other agent's nearest  body point
            state.append(findSnakeDirection(agent))   # direction of the other agent
            state.append(relativePoints(snake.head,findOtherSnakeNearestPoint(agent, snake)))  # nearest body point of the snake to the other agent's head
        else:
            state.extend([Point(-1, -1), Point(-1, -1), -1, Point(-1, -1)])

    state.append(relativePoints(snake.head,findNearestWall(snake)))  # nearest wall point

    return state

''' Returns the length of the state according to if the game is single or
multiple snake game '''
def getStateLength():
    if Constants.existsMultipleAgents == False:
        return 9
    elif Constants.existsMultipleAgents == True:
        return 3 + (Constants.numNearestFoodPointsForState*2) + (Constants.numberOfSnakes-1)*7

''' This method is called with the arguments that specify if its a
multiagent setting, if relative or absolute state space has to be used,
if normalisation has to be applied, along with the other arguments'''
def getState(snake, agentList, food, normalize=False):
    state = []
    if snake.alive == False:
        return [-1] * getStateLength()
    if Constants.useRelativeState == False and Constants.existsMultipleAgents == False:
        state.extend(getAbsoluteStateForSingleAgent(snake, food))
    elif Constants.useRelativeState == False and Constants.existsMultipleAgents == True:
        state.extend(getAbsoluteStateForMultipleAgents(snake, agentList, food))
    elif Constants.useRelativeState == True and Constants.existsMultipleAgents == False:
        state.extend(getRelativeStateForSingleAgent(snake, food))
    elif Constants.useRelativeState == True and Constants.existsMultipleAgents == True:
        state.extend(getRelativeStateForMultipleAgents(snake, agentList, food))

    flatState = []
    for entry in state:
        if isinstance(entry, Point):
            if normalize:
                flatState.append( entry.x * 1.0 / Constants.gridSize )
                flatState.append( entry.y * 1.0 / Constants.gridSize )
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
