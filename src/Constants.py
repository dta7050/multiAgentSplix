""" This file contains the global constants
which are used across the project"""

# Constants relevant for the entire game
globalEpisodeLength = 300  # Length of each learning/training episode, Type: int
numberOfSnakes = 1  # Default number of snakes in each game, Type: int
maximumFood = 10  # Maximum number of food points on the screen, Type: int
gridSize = 30  # Size of game window, Type: int

# Constants pertaining to State
numNearestFoodPointsForState = 3  # Number of nearest food points the agent is aware of, Type: int
useRelativeState = False  # Whether agent uses relative information of other agents, Type: bool
existsMultipleAgents = numberOfSnakes > 1  # Whether or not the are multiple agents, Type: bool

# Constants general to both algorithms
gamma = 1.0  # Discount factor for reinforcement learning algorithms, type: float

# Constants for ActorCritic algorithm
AC_alphaW = 0.0022  # Constant used to update learning parameters in actor-critic method, Type: float
AC_alphaTheta = 0.0011  # Constant used to update theta values in actor-critic method, Type: float

# Constants for AsynchronousQ algorithm
AQ_lr = 0.0001  # Learning rate for Q-learning method, Type: float
AQ_asyncUpdateFrequency = 128  # Determines how often agent policy is updated, Type: int
AQ_globalUpdateFrequency = 512  # Determines how often checkpoints are created in learning process, Type: int
