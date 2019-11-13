''' This file contains the Game class. The object of this class is instantiated
when each new game is created. It initialises the game with the creation of snakes,
grid and the initial food points. It also contains a method to return the single
stage reward and to indicate if the episode has ended or not '''

import numpy as np

import Constants

from Snake import Snake
from Food import Food
from Action import Action
from typing import List


class Game:
    def __init__(self, numOfSnakes: int =Constants.numberOfSnakes, gridSize: int =Constants.gridSize,
                 maxEpisodeLength: int =Constants.globalEpisodeLength):
        """
        Initializes the game with the number of snakes, size of game window, pieces of food, and maximum length
        of game
        :param numOfSnakes: number of snake to initialize
        :param gridSize: size of the game window
        :param maxEpisodeLength: maximum length of time the game lasts
        """
        self.snakes: list = []  # empty list of snakes
        for idx in range(numOfSnakes):
            self.snakes.append(Snake(idx))  # adds snakes to the list

        self.food = Food(Snake.snakeList)  # initializes the food points on screen? (can't find declaration of Food())
        self.gameLength: int = maxEpisodeLength
        self.time_step: int = 0  # int to keep track of the length of the game
        return

    def __str__(self) -> str:
        """
        Prints a message displaying the status of the game (amount of time steps, food, and snakes)
        :return: returns the message to be printed
        """
        print_message = "Time " + str(self.time_step) + "\n"  # adds the amount of time steps to the message
        print_message += "Food = " + str(map(str, self.food.foodList)) + "\n"  # adds location of food to the message
        for s in self.snakes:
            print_message += str(s) + "\n"  # adds number of snakes to message
        return print_message

    def move(self, actionsList: list = []) -> (List[int], bool):
        """
        Takes in a list of actions corresponding to each snake in the game.
        If a snake is dead, then its corresponding position in actionsList simply holds None.
        Returns boolean indicating whether the game has ended

         The Game ends if the time step has reached the specified length of the game, or if all snakes are dead.

        :param actionsList: list of potential actions snake can take
        :return:
        """
        assert len(actionsList) == len(self.snakes), "Deficiency of actions provided."

        action_validity_check = []
        # sifts through list of possible actions of each snake and throws an error if there are any invalid ones
        for i in range(len(self.snakes)):
            s = self.snakes[i]
            if s.alive:
                permissible_actions = s.permissible_actions()
                action_validity_check.append(actionsList[i] in permissible_actions)
        assert all(action_validity_check), "At least one action is invalid"

        self.time_step += 1  # increment time steps

        single_step_rewards = [0]*len(self.snakes)  # gives list of zeros with length equal to the number of snakes
        """
        Increments through each snake and their corresponding actions and performs
        necessary actions
        """
        for i in range(len(actionsList)):
            snake = self.snakes[i]
            if snake.alive:
                snake.moveInDirection(actionsList[i])  # moves snake based on chosen action
                # if the snake ate food, grow snake, destroy food, and reward agent
                if snake.didEatFood(self.food):
                    snake.incrementScore(10)
                    snake.growSnake()
                    self.food.eatFood(snake.head, Snake.snakeList)
                    single_step_rewards[i] = 10
                    # if the snake hit a wall, kill snake and punish agent
                elif snake.didHitWall():
                    snake.backtrack()
                    snake.killSnake(self.food)
                    snake.incrementScore(-10)
                    single_step_rewards[i] = -10

        # iterates through each snake to see if two snakes hit each other
        for i in range(len(self.snakes)):
            for j in range(len(self.snakes)):
                if i == j:
                    continue
                if not (self.snakes[i].alive and self.snakes[j].alive):
                    continue
                # if one snake hits another, kill the snake (not clear which one)
                if self.snakes[i].didHitSnake(self.snakes[j]):
                    self.snakes[i].backtrack()
                    self.snakes[i].killSnake(self.food)

        return (single_step_rewards, not(self.time_step == self.gameLength or all([not s.alive for s in self.snakes])))

    def endGame(self) -> list:
        """
        Deletes food and snakes, returns final scores
        :return: list of each snakes final score
        """
        scoreboard: list = [s.score for s in self.snakes]
        del self.food  # deletes the food on screen
        del self.snakes  # deletes the snakes on the screen
        return scoreboard
