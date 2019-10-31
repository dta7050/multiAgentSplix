''' This file contains the global constants
which are used across the project'''

# Constants relevant for the entire game
globalEpisodeLength = 300
numberOfSnakes = 3
maximumFood = 10
gridSize = 30

# Constants pertaining to State
numNearestFoodPointsForState = 3
useRelativeState = False
existsMultipleAgents = numberOfSnakes > 1

# Constants general to both algorithms
gamma=1.0

# Constants for ActorCritic algorithm
AC_alphaW = 0.0022
AC_alphaTheta=0.0011

# Constants for AsynchronousQ algorithm
AQ_lr = 0.0001
AQ_asyncUpdateFrequency = 128
AQ_globalUpdateFrequency = 512
