''' This file contains the Snake class which creates the snake objects.
It ensures that the snakes spawned do not overlap. Each snake has a head, tail,
joints, 'id', score, alive/dead. It also has methods for getting the body of the
snake, method that returns a boolean indicating if the snake has eaten food, if
the snake hit the wall or another snake, to update the snake's object according
to the movement of the snake in a particular direction, return the permissible
actions for a snake, and other helper methods to maintain the snake object'''

import Constants

from numpy.random import randint

from Point import Point
from Action import Action
from Food import Food
from typing import List, Tuple


class Snake:
    snakeList = []  # contains all snakes created for the game

    def __init__(self, identity: int):
        """
        Initializes the Snake object
        :param identity: integer number that identifies a Snake object
        """
        # Get all points of current snakes in game, store them in occupiedPoints list
        occupiedPoints = []  # type: List[Point]
        for snake in Snake.snakeList:
            body = snake.getBodyList()
            bodyPoints = Point.returnBodyPoints(body)  # type: List[Point]
            occupiedPoints.extend(bodyPoints)

        # create the body of this snake instance
        while True:
            # generate point with at least 10 units gap from any wall
            self.head = Point(randint(10, Constants.gridSize - 10), randint(10, Constants.gridSize - 10))
            self.end = Point(self.head.x - 5, self.head.y)
            self.joints = []
            # First, the body is made up of a head point, end point, and joints (none yet)
            body = self.getBodyList()
            # Next, the returnBodyPoints function calculates all intermediate points of the snake
            bodyPoints = Point.returnBodyPoints(body)
            # If the generated body list does not contain any "occupiedPoints" break this loop;
            # otherwise generate a new body. (This could be improved)
            if not bool(self.lists_overlap(bodyPoints, occupiedPoints)):
                break

        self.id = identity  # type: int  # assigns id number to snake
        self.alive = True   # tells whether the snake is alive or not (True = alive)
        self.score = 0      # contains the score of the snake
        # the following attributes contain the previous positional state of the snake
        self.prev_head, self.prev_joints, self.prev_end = self._copy(self.head, self.joints, self.end)
        # add Snake object to list containing all created snakes
        Snake.snakeList.append(self)
        return

    def __str__(self) -> str:
        """
        Returns status message of Snake containing life status, position and score
        :return: str, the status message
        """
        if self.alive:
            body = self.getBodyList()
            body_str = str(map(str, body))
            print_message = "Snake {}\n\tPosition = {}\n\tScore = {}\n".format(self.id, body_str, self.score)
            return print_message
        else:
            print_message = "Snake {}\n\tDead. Score = {}\n".format(self.id, self.score)
            return print_message

    ''' Helper method to check if the bodies of the snake overlap '''
    def lists_overlap(self, list1: List[Point], list2: List[Point]) -> bool:
        """
        Returns True if list1 and list2 contain any equivalent Points, returns False otherwise
        :param list1: first list to compare
        :param list2: second list to compare
        :return: bool
        """
        for point in list1:
            if point in list2:
                return True
        return False

    ''' Returns the body of the snake stitching together the head, tail and the joints '''
    def getBodyList(self) -> List[Point]:
        """
        Returns list only containing the head, end, and joint points of a Snake.
        NOTE: Does not return connecting points!
        :return: List[Point]
        """
        body = [self.head]
        body.extend(self.joints)
        body.append(self.end)
        return body

    ''' Returns a boolean indicating if a food point has been eaten by a snake '''
    # TODO: WE WILL NOT NEED THIS FUNCTION FOR OUR GAME !!
    def didEatFood(self, food) -> bool:
        return (self.head in food.foodList)

    ''' Increments the score of a snake '''
    def incrementScore(self, reward: int) -> None:
        """
        Adds a reward to the Snake's score
        :param reward: integer value corresponding to the reward to add to the score
        :return: None
        """
        self.score += reward
        return

    ''' Returns a boolean indicating if the snake hit a wall '''
    def didHitWall(self) -> bool:
        """
        Returns True if the Snakes head hit a wall
        Returns False otherwise
        :return: bool
        """
        return (self.head.x == 0 or self.head.x == Constants.gridSize or
                self.head.y == 0 or self.head.y == Constants.gridSize)

    ''' Method to update the point coordinates according to the direction
    of movement '''
    def _update_point(self, p: Point, direction: int) -> Point:
        """
        Updates a point's coordinates depending on the Action the Snake is taking
        :param p: Point object to move
        :param direction: integer corresponding to the direction to move p in
        :return: the moved point, Point
        """
        if direction == Action.TOP:
            p.y += 1
        elif direction == Action.DOWN:
            p.y -= 1
        elif direction == Action.RIGHT:
            p.x += 1
        elif direction == Action.LEFT:
            p.x -= 1
        return p

    def _copy(self, head, joints, end) -> Tuple[Point, List[Point], Point]:
        """
        Returns a copy of the Snakes head, joints and endpoint
        :param head: Point corresponding to head of Snake
        :param joints: Points corresponding to joints of Snake
        :param end: Point corresponding to endpoint of Snake
        :return: Tuple containing (Head, Joints, End)
        """
        _head = Point.fromPoint(head)
        _joints = [Point.fromPoint(p) for p in joints]
        _end = Point.fromPoint(end)
        return _head, _joints, _end

    def backtrack(self) -> None:
        """
        Sets the current head, joints, an end points of a Snake to its previous head, joints and end points
        :return: None
        """
        self.head, self.joints, self.end = self._copy(self.prev_head, self.prev_joints, self.prev_end)
        return

    ''' Moves the snake in a direction chosen by it at each time step. This updates
    the snake body points '''
    def moveInDirection(self, action: Action) -> None:
        """
        Adjusts the Snakes head, joints and end point given an Action to move in a direction
        :param action: Action for Snake to take
        :return: None
        """
        # First, check to see if we can make the given action
        assert (action in self.permissible_actions()), "Action not allowed in this state."
        # Store current state of snake as the new previous state of snake before taking action
        self.prev_head, self.prev_joints, self.prev_end = self._copy(self.head, self.joints, self.end)

        # move the snake in the direction specified
        if self.joints == []:  # Then Snake is a straight line horizontal or vertical
            # get current direction the Snake is heading
            direction = self.findDirection(self.head, self.end)
            if direction != action:  # add joint when snake changes direction
                self.joints.append(Point.fromPoint(self.head))
            # Update head and end points
            self.head = self._update_point(self.head, action)
            self.end = self._update_point(self.end, direction)
        else:  # Then Snake can be any shape with any number of joints
            # get current direction the Snake is heading
            direction = self.findDirection(self.head, self.joints[0])
            if direction != action: # add joint when snake changes direction
                self.joints.insert(0, Point.fromPoint(self.head))
            self.head = self._update_point(self.head, action)
            direction = self.findDirection(self.joints[-1], self.end)
            self.end = self._update_point(self.end, direction)
            if (self.end.x == self.joints[-1].x) and (self.end.y == self.joints[-1].y):  # pop joint if end reached it
                self.joints = self.joints[:-1]

        return

    ''' Returns a boolean indicating if the snake hit another snake '''
    def didHitSnake(self, opponent_snake: 'Snake') -> bool:
        """
        If this Snake has collided with 'opponent_snake' return True.
        Else, return False.
        :param opponent_snake: Other Snake to check collision with
        :return: bool
        """
        # get head, joints, and end of opponent snake
        body = opponent_snake.getBodyList()  # type: List[Point]

        p = self.head  # type: Point
        # If Snake's heads collide...
        if p == opponent_snake.head:
            # TODO: CHANGE THIS DEPENDING ON OUR RULE
            # if heads collide, the larger snake remains. If the scores are equal, then both snakes should die.
            return self.score <= opponent_snake.score

        for i in range(len(body)-1):
            p1 = body[i]  # point of opponent snake
            p2 = body[i+1]  # next point of opponent snake
            if p1.x == p2.x:  # points make up a vertical line of opponent snake
                if p.x == p1.x:  # then head of snake is at the same x coord of opponent
                    # check if snake head has collided with opponent snake points
                    lim1, lim2 = tuple(sorted([p1.y, p2.y]))
                    if p.y in range(lim1, lim2 + 1):
                        return True
            else:  # points make up a horizontal line of opponent snake
                if p.y == p1.y:  # then head of snake is at the same y coord of opponent
                    # check if snake head has collided with opponent snake points
                    lim1, lim2 = tuple(sorted([p1.x, p2.x]))
                    if p.x in range(lim1, lim2 + 1):
                        return True
        return False

    ''' Returns the direction of P1 with reference to P2 '''
    def findDirection(self, p1: Point, p2: Point) -> int:
        """
        Returns the direction of P1 with reference to P2
        :param p1: Point to be compared to
        :param p2: Point to compare with p1
        :return: integer value corresponding to direction
        """
        if p1.x - p2.x == 0 and p1.y - p2.y < 0:
            return Action.DOWN
        elif p1.x - p2.x == 0 and p1.y - p2.y > 0:
            return Action.TOP
        elif p1.x - p2.x > 0 and p1.y - p2.y == 0:
            return Action.RIGHT
        elif p1.x - p2.x < 0 and p1.y - p2.y == 0:
            return Action.LEFT

    ''' Grows the snake in the direction of the tail once the food is eaten.
    Should be called before moveInDirection '''
    # TODO: WE WILL NOT NEED THIS IF WE CHANGE THE GAME
    def growSnake(self):
        if self.joints == []:
            direction = self.findDirection(self.end, self.head)
         # Finding direction from the last joint/head to tail as it is in this direction the increment should happen
        else:
            direction = self.findDirection(self.end, self.joints[-1])
        self.end = self._update_point(self.end, direction)

    def permissible_actions(self) -> List[Action]:
        """
        Returns the permissible actions for a snake, given the direction of motion of the snake
        :return: list of actions the snake can take, List[Action]
        """
        actions = []  # type: List[Action]
        if self.joints == []:  # Then snake is just a vertical or horizontal line
            direction = self.findDirection(self.end, self.head)
        else:
            direction = self.findDirection(self.joints[0], self.head)
        # Permissible actions are those that move the Snake not back the same way it is coming from
        for act in (Action):
            if act != direction:
                actions.append(act)
        return actions

    # TODO: WILL NEED TO CHANGE IF WE DO SPLIX GAME
    def killSnake(self, food: Food) -> None:
        """
        Once the snake is dead, this method sets the alive bit of the snake to false,
        converts the body points into food points and deletes the snake's head, joints and tail
        :param food:
        :return: None
        """
        self.alive = False  # kill Snake

        # Convert body points into Food
        body = self.getBodyList()
        points = Point.returnBodyPoints(body)
        food.addFoodToList(points)

        # Delete Snake
        Snake.snakeList.remove(self)
        del self.head
        del self.end
        del self.joints
