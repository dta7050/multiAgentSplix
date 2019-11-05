''' This file contains the Food class which has
methods that create food points, add new food
points created to the foodlist, and remove the
food point from the foodlist once the point is eaten '''

import Constants

from numpy.random import randint
from typing import List

from Point import Point
from Snake import Snake


class Food:

    def __init__(self, snakes: List[Snake] = []):
        """
        Initializes the Food object
        :param snakes: list of Snake objects
        """
        self.foodList = []  # type: List[Point]  # Contains all points that are food in the game
        self.createFood(Constants.maximumFood, snakes)

    def createFood(self, n: int, snakes: List[Snake] = []) -> None:
        """
        This method spawns specified number of food points in the grid at random positions.
        It also ensures that the food points created are not overlapping
        :param n: number of food points to make
        :param snakes: List of all Snakes in game. Used to create list of occupied points so that food points
        do not get placed on top of snakes
        :return: None
        """
        # We dont want to place food where Snake bodies already are. This gets a list of all occupied points
        occupied_points = []  # type: List[Point]
        for snake in snakes:
            body = snake.getBodyList()
            bodyPoints = Point.returnBodyPoints(body)
            occupied_points.extend(bodyPoints)

        # Generate random food points
        for i in range(n):
            while True:
                x = randint(1, Constants.gridSize-1)
                y = randint(1, Constants.gridSize-1)
                p = Point(x,y)
                if p not in occupied_points and p not in self.foodList:
                    self.foodList.append(p)
                    break

    def addFoodToList(self, pointList: List[Point]) -> None:
        """
        This method is used to update the food list either when new food points get added
        or when the dead snake's body points are converted to food points
        :param pointList: list of food points
        :return: None
        """
        for p in pointList:
            self.foodList.append(p)

    def eatFood(self, food_point: Point, snakes: List[Snake] = []) -> None:
        """
        This method deletes the food from the food list once it has been eaten.
        It also maintains a minimum number of food points in the grid at an instance
        :param food_point: Point that is a food point that will be eaten
        :param snakes: List of all Snakes in game
        :return: None
        """
        # Remove the eaten food point from food list
        for i, f in enumerate(self.foodList):
            if f == food_point:
                del self.foodList[i]

        # Add a new food point to the game to replace it
        if len(self.foodList) < Constants.maximumFood:
            self.createFood(Constants.maximumFood - len(self.foodList), snakes)
