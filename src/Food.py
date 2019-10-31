''' This file contains the Food class which has
methods that create food points, add new food
points created to the foodlist, and remove the
food point from the foodlist once the point is eaten '''

import Constants

from numpy.random import randint

from Point import Point


class Food:

    def __init__(self, snakes=[]):
        self.foodList = []
        self.createFood(Constants.maximumFood, snakes)

    ''' This method spawns specified number of food points in the
    grid at random positions. It also ensures that the food points
    created are not overlapping '''
    def createFood(self, n, snakes=[]):
        occupiedPoints = []
        for snake in snakes:
            body = snake.getBodyList()
            bodyPoints = Point.returnBodyPoints(body)
            occupiedPoints.extend(bodyPoints)

        for i in range(n):
            while True:
                x = randint(1, Constants.gridSize-1)
                y = randint(1, Constants.gridSize-1)
                p = Point(x,y)
                if p not in occupiedPoints and p not in self.foodList:
                    self.foodList.append(p)
                    break

    ''' This method is used to update the foodlist either when new
    food points get added  or when the dead snake's body points are
    converted to food points '''
    def addFoodToList(self, pointList):
        for p in pointList:
            self.foodList.append(p)

    ''' This method deletes the food from the foodlist once it has
    been eaten. It also maintains a minimum number of food points in
    the grid at an instance '''
    def eatFood(self, food, snakes=[]):
        for i, f in enumerate(self.foodList):
            if f == food:
                del self.foodList[i]

        if(len(self.foodList) < Constants.maximumFood):
            self.createFood(Constants.maximumFood - len(self.foodList), snakes)
