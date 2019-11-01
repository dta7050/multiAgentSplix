''' This file contains the point class which defines the (x, y)
coordinates of a point. It also contains methods to return all
the body points of a snake, compare the equality of two points
and check if an object is an instance of the point class '''

import copy
from typing import List


class Point:
    def __init__(self, x: int, y: int):
        """
        Initializes the Point object with its coordinates
        :param x: x-coordinate of Point
        :param y: y-coordinate of Point
        """
        self.x = x
        self.y = y

    ''' This method compares if an object is an instance of Point
    class '''
    @classmethod
    def fromPoint(cls, p) -> Point:
        """
        Sees if argument 'p' is a Point object
        :param p: object that we are checking to see if it is a Point object
        :return: returns a Point instance
        """
        assert isinstance(p, Point), "Invalid parameter passed"
        return cls(p.x, p.y)  # returns a Point instance with the x, y coords of p (essentially returns back the point)

    def __str__(self) -> str:
        """
        Creates a string based off of a Point object that displays its coordinates
        :return:
        """
        return "({}, {})".format(self.x, self.y)

    ''' This method is used to compare the equality of two points '''
    def __eq__(self, p) -> bool:
        """
        Checks if two points are equal
        :param p: Point that you are comparing to
        :return: boolean, True if Points are equal; False if not
        """
        return self.x == p.x and self.y == p.y

    ''' This method takes in the head, joints and tail of a snake
    and returns all the points along the body of the snake '''
    @staticmethod
    def returnBodyPoints(body: List[Point]) -> List[Point]:
        """
        Constructs a list of all Points of a Snake given its body
        :param body: contains the head, joints and tail of a Snake
        :return: points (List[Point])
        """
        points = copy.deepcopy(body)  # type: List[Point]
        # iterate through all Points in body
        for i in range(len(body) - 1):
            p1 = body[i]
            p2 = body[i + 1]
            if p1.x == p2.x:    # vertical line
                # add vertical line points inbetween p1 and p2
                for y in range(min(p1.y, p2.y) + 1, max(p1.y, p2.y)):
                    points.append(Point(p1.x, y))
            else:   # horizontal line
                # add horizontal line points inbetween p1 and p2
                for x in range(min(p1.x, p2.x) + 1, max(p1.x, p2.x)):
                    points.append(Point(x, p1.y))
        return points
