# Multi-Agent-Snake

Main.py: This file is the entry point to the project. It takes in the user arguments, processes it and then calls the appropriate functions accordingly

Action.py: This file contains an enum class for the 4 actions top, down, left and right

Constants.py: This file contains the global constants which are used across the project

Food.py: This file contains the Food class which has methods that create food points, add new food points created to the food-list, and remove the food point from the food-list once the point is eaten

Point.py: This file contains the point class which defines the (x, y) coordinates of a point. It also contains methods to return all the body points of a snake, compare the equality of two points and check if an object is an instance of the point class.

Agent.py: This file contains methods to compute the state space for a given snake. 'getState' method is called with the arguments that specify if its a multiagent setting, if relative or absolute state space has to be used, if normalisation has to be applied, along with the other arguments. The state space is as it is described in section 3.2.

Snake.py: This file contains the Snake class which creates the snake objects. It ensures that the snakes spawned do not overlap. Each snake has a head, tail, joints, 'id', score, alive/dead. It also has methods for getting the body of the snake, method that returns a boolean indicating if the snake has eaten food, if the snake hit the wall or another snake, to update the snake's object according to the movement of the snake in a particular direction, return the permissible actions for a snake, and other helper methods to maintain the snake object.

ActorCritic.py: This file contains the implementation of the actor critic algorithm. It contains helper methods to get policy and feature vector that are necessary for the algorithm. It also contains a method to run the game on a graphical user interface once the agent has been trained.

Game.py: This file contains the Game class. The object of this class is instantiated when each new game is created. It initialises the game with the creation of snakes, grid and the initial food points. It also contains a method to return the single stage reward and to indicate if the episode has ended or not


Guide to run the project

The agent in the game can be trained by running the following command:
Python main.py --mode=train --algorithm=asyncQ/actorcritic --train_time_steps=3000
The game can be simulated(run the game after the agents have been trained with the algorithms used) with the following command:
Python main.py --mode=simulate --algorithm=asyncQ/actorcritic --trained_ckpt_index=3000
