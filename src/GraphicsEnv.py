"""This file contains the graphics/UI environment for the snake game
Is used to represent both the multi-agent and the single agent snake game
User can also manually play either a single-snake game or multi-snake game
along with the agents"""

import pygame
import math
import random
import numpy as np

import Game
import Food
import Constants
import Point
import Snake

from Action import Action
from pygame import Surface, Event, display


pygame.init()  # Initializes all pygame modules

"""Used to draw the score"""


def draw_text(surf: Surface, text: str, size: int, x: int, y: int, color: tuple):
    """
    Function to draw text on the game screen
    :param surf: a pygame surface for the text to be printed on
    :param text: string of text to be displayed
    :param size: font size of the text
    :param x: x coordinate of the text
    :param y: y coordinate of the text
    :param color: tuple containing RGB color values of text
    :return: function doesn't return anything
    """
    font_name = pygame.font.match_font('arial')
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)


"""Convert coordinates into pygame coordinates (lower-left => top left)."""
def to_pygame(p: Point) -> int:

    return (p.x, Constants.gridSize-p.y)


def manual_action_list(g: Game, event: Event) -> list:
    """
    Function to determine where to move each snake. Iterates through each snake assigning an action to each one. If the
    user is playing the game, it reads the button presses and assigns an action accordingly
    :param g: An object of class Game. Stores information about the current state of the game
    :param event: Some kind of game event such as a button press. Used to identify if a button was pressed
    :return: A list of each snake's next action
    """
    """
    This initialization may be a bit computation intensive, but added this line here for representation
    of two or more snakes
    Else can simply initialize to 0
    """
    actionsList = [0]*Constants.numberOfSnakes
    for i, snake in enumerate(g.snakes):  # iterates through each snake
        if not snake.alive:
            actionsList[i] = None
        else:
            actionsList[i] = random.choice(snake.permissible_actions())  # assigns a random action to the snake

    if g.snakes[0].alive:   # user's snake
        keys = pygame.key.get_pressed()  # To get the keys pressed by user
        if g.snakes[0].joints == []:
            defaultaction = g.snakes[0].findDirection(g.snakes[0].head, g.snakes[0].end)
        else:
            defaultaction = g.snakes[0].findDirection(g.snakes[0].head, g.snakes[0].joints[0])

        actionsList[0] = defaultaction
        user_permissible_actions = g.snakes[0].permissible_actions()
        if event.type == pygame.KEYDOWN:  # assigns the appropriate action based on key press
            if keys[pygame.K_RIGHT] and Action.RIGHT in user_permissible_actions:
                actionsList[0] = Action.RIGHT
            elif keys[pygame.K_LEFT] and Action.LEFT in user_permissible_actions:
                actionsList[0] = Action.LEFT
            elif keys[pygame.K_UP] and Action.TOP in user_permissible_actions:
                actionsList[0] = Action.TOP
            elif keys[pygame.K_DOWN] and Action.DOWN in user_permissible_actions:
                actionsList[0] = Action.DOWN

    return actionsList


def manual_action(snake: Snake, event: Event) -> Action:
    """
    A function that determines the action to be taken by the player's snake based on the button presses detected
    by the event variable
    :param snake: A reference to the user's snake
    :param event: Some kind of game event such as a button press. Used to identify if a button was pressed
    :return: Returns the action to be taken
    """
    if snake.alive: # user's snake
        keys = pygame.key.get_pressed()  # get the button press
        if snake.joints == []:  # default action: keep snake moving forward
            defaultaction = snake.findDirection(snake.head, snake.end)
        else:
            defaultaction = snake.findDirection(snake.head, snake.joints[0])

        action_taken = defaultaction
        user_permissible_actions = snake.permissible_actions()
        if event.type == pygame.KEYDOWN:  # if a button was pressed, change the action to be taken
            if keys[pygame.K_RIGHT] and Action.RIGHT in user_permissible_actions:
                action_taken = Action.RIGHT
            elif keys[pygame.K_LEFT] and Action.LEFT in user_permissible_actions:
                action_taken = Action.LEFT
            elif keys[pygame.K_UP] and Action.TOP in user_permissible_actions:
                action_taken = Action.TOP
            elif keys[pygame.K_DOWN] and Action.DOWN in user_permissible_actions:
                action_taken = Action.DOWN
    else:
        action_taken = None  # if the snake is dead, no action is taken
    return action_taken


"""This function is used to draw the body of the snake, display score, display food points
for a manual game of snake"""

def runRandomGame(play: bool = True, scalingFactor: int = 9):  # Scaling the size of the grid):
    """
    Function initializes the game window, computes the next action of each snake, performs those actions, and
    updates the game window for each frame
    :param play: boolean variable indicating if the user is to play with the learning agents
    :param scalingFactor: integer used to scale the size of the game window
    :return: null
    """
    g = Game.Game()  # Instantiating an object of class Game
    width = scalingFactor * Constants.gridSize  # sets size of game window
    height = scalingFactor * Constants.gridSize
    pos_score_x = int(math.floor(width / (Constants.numberOfSnakes + 1)))
    pos_score_y = int(math.floor(height / 20))  # determines where on screen to display scores
    font_size = int(math.floor(height / 40))  # determines size of the font
    black = (0, 0, 0)  # initialize four variable holding RGB color values
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    colors = np.random.randint(0, 256, size=[Constants.numberOfSnakes, 3])  # creates a random color
    if play:  # user interacts with the agents
        colors[0] = black  # player's snake is always black
    crashed = False  # boolean to indicate a crash of the game
    episodeRunning = True  # boolean to show if a game episode is in progress
    # initializes the game window
    win = pygame.display.set_mode((scalingFactor * Constants.gridSize, scalingFactor * Constants.gridSize))
    screen = pygame.Surface((Constants.gridSize+1, Constants.gridSize+1))  # initializes the play screen
    pygame.display.set_caption("Snake Game")
    clock = pygame.time.Clock()

    while not crashed and episodeRunning:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # check if the game has ended
                crashed = True
        screen.fill(white)

        pygame.draw.lines(screen, black, True, [(0, 0), (0, Constants.gridSize),
                                                (Constants.gridSize, Constants.gridSize),
                                                (Constants.gridSize, 0)])  # draw the walls

        # The below loop draws all the food particles as points.
        for p in g.food.foodList:
            pygame.draw.line(screen, green, to_pygame(p), to_pygame(p), 1)  # Drawing all the food points

        # This is for drawing the snake and also the snake's head is colored red
        for idx in range(Constants.numberOfSnakes):
            if g.snakes[idx].alive:
                body = g.snakes[idx].getBodyList()
                for i in range(len(body) - 1):
                    pygame.draw.line(screen, colors[idx], to_pygame(body[i]), to_pygame(body[i + 1]), 1)
                pygame.draw.line(screen, red, to_pygame(body[0]), to_pygame(body[0]), 1)

        actionsList = manual_action_list(g, event)  # generate actions for each snake
        """
        actionsList = rl_agent(g)
        Can also add an if condition here to have one player and one agent
        and then append the two lists obtained, to pass it to move method
        """
        _, episodeRunning = g.move(actionsList)  # performs actions in actionsList, returns rewards and a bool
        # Transforms the screen window into the win window
        win.blit(pygame.transform.scale(screen, win.get_rect().size), (0, 0))
        for idx in range(Constants.numberOfSnakes):
            draw_text(win, "Snake" + str(idx) + "  " + str(g.snakes[idx].score), font_size,
                      pos_score_x * (idx + 1), pos_score_y, black)  # Displays the score of each snake
        pygame.display.update()  # update the display
        clock.tick(10)  # (FPS)means that for every second at most 10 frames should pass.
    pygame.quit()  # end the game


def displayGame(game: Game, win: display, screen: Surface, colors: list[tuple], scalingFactor: int = 9):
    """
    A function to display a game episode when the player. It takes in a window and screen/surface and
    updates the display after each frame based on the actions taken
    :param game: An object of class Game. Stores information about the current state of the game
    :param win: A window for the game to be displayed on
    :param screen: A pygame surface that will be used to draw the game on the window
    :param colors: A list of colors so each snake is a different color
    :param scalingFactor: An integer used to scale the size of the game window
    :return: null
    """
    width = scalingFactor * Constants.gridSize  # determines size of game window
    height = scalingFactor * Constants.gridSize
    # determines where each snake's score is displayed
    pos_score_x = int(math.floor(width / (Constants.numberOfSnakes + 1)))
    pos_score_y = int(math.floor(height / 20))
    font_size = int(math.floor(height / 40))  # determines size of the font
    # a few colors pre-initialized
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)

    clock = pygame.time.Clock()
    screen.fill(white)
    # draw the walls
    pygame.draw.lines(screen, black, True, [(0, 0), (0, Constants.gridSize),
                                            (Constants.gridSize, Constants.gridSize),
                                            (Constants.gridSize, 0)])  # sets the boundaries of the game

    # The below loop draws all the food particles as points.
    for p in game.food.foodList:
        pygame.draw.line(screen, green, to_pygame(p), to_pygame(p), 1)  # Drawing all the food points

    # This is for drawing the snake and also the snake's head is colored red
    for idx in range(len(game.snakes)):
        if game.snakes[idx].alive:
            body = game.snakes[idx].getBodyList()
            for i in range(len(body) - 1):
                pygame.draw.line(screen, colors[idx], to_pygame(body[i]), to_pygame(body[i + 1]), 1)
            pygame.draw.line(screen, red, to_pygame(body[0]), to_pygame(body[0]), 1)

    # Transforms the screen window into the win window
    win.blit(pygame.transform.scale(screen, win.get_rect().size), (0, 0))
    for idx in range(len(game.snakes)):
        draw_text(win, "Snake" + str(idx) + "  " + str(game.snakes[idx].score),
                  font_size, pos_score_x * (idx + 1), pos_score_y, black)  # displays the score of each snake
    pygame.display.update()  # updates the display after the actions take place
    clock.tick(10)  # (FPS)means that for every second at most 10 frames should pass.
    return


if __name__ == '__main__':
    runRandomGame()
