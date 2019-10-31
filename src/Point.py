''' This file contains the point class which defines the (x, y)
coordinates of a point. It also contains methods to return all 
the body points of a snake, compare the equality of two points
and check if an object is an instance of the point class '''

import copy

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    ''' This method compares if an object is an instance of Point
    class '''
    @classmethod
    def fromPoint(cls, p):
        assert isinstance(p, Point), "Invalid parameter passed"
        return cls(p.x, p.y)

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    ''' This method is used to compare the equality of two points '''
    def __eq__(self, p):
        return self.x == p.x and self.y == p.y

    ''' This method takes in the head, joints and tail of a snake
    and returns all the points along the body of the snake '''
    @staticmethod
    def returnBodyPoints(body):
        points = copy.deepcopy(body)
        for i in range(len(body) - 1):
            p1 = body[i]
            p2 = body[i + 1]
            if p1.x == p2.x:    # vertical line
                for y in range(min(p1.y, p2.y) + 1, max(p1.y, p2.y)):
                    points.append(Point(p1.x, y))
            else:   # horizontal line
                for x in range(min(p1.x, p2.x) + 1, max(p1.x, p2.x)):
                    points.append(Point(x, p1.y))

        return points
