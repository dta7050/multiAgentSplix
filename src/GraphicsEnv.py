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

from Action import Action


pygame.init()  # Initializes all pygame modules

"""Used to draw the score"""

def draw_text(surf, text, size,x,y,color):
    font_name = pygame.font.match_font('arial')
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface,text_rect)

"""Convert coordinates into pygame coordinates (lower-left => top left)."""
def to_pygame(p):

    return (p.x, Constants.gridSize-p.y)


def manual_action_list(g,event):
    """
    This initialization may be a bit computation intensive, but added this line here for representation of two or more snakes
    Else can simply initialize to 0
    """
    actionsList = [0]*Constants.numberOfSnakes
    for i, snake in enumerate(g.snakes):
        if not snake.alive:
            actionsList[i] = None
        else:
            actionsList[i] = random.choice(snake.permissible_actions())

    if g.snakes[0].alive: # user's snake
        keys = pygame.key.get_pressed() #To get the keys pressed by user
        if g.snakes[0].joints == []:
            defaultaction = g.snakes[0].findDirection(g.snakes[0].head, g.snakes[0].end)
        else:
            defaultaction = g.snakes[0].findDirection(g.snakes[0].head, g.snakes[0].joints[0])

        actionsList[0] = defaultaction
        user_permissible_actions = g.snakes[0].permissible_actions()
        if event.type == pygame.KEYDOWN:
            if keys[pygame.K_RIGHT] and Action.RIGHT in user_permissible_actions:
                actionsList[0] = Action.RIGHT
            elif keys[pygame.K_LEFT] and Action.LEFT in user_permissible_actions:
                actionsList[0] = Action.LEFT
            elif keys[pygame.K_UP] and Action.TOP in user_permissible_actions:
                actionsList[0] = Action.TOP
            elif keys[pygame.K_DOWN] and Action.DOWN in user_permissible_actions:
                actionsList[0] = Action.DOWN

    return actionsList

def manual_action(snake, event):
    if snake.alive: # user's snake
        keys = pygame.key.get_pressed()
        if snake.joints == []:
            defaultaction = snake.findDirection(snake.head, snake.end)
        else:
            defaultaction = snake.findDirection(snake.head, snake.joints[0])

        action_taken = defaultaction
        user_permissible_actions = snake.permissible_actions()
        if event.type == pygame.KEYDOWN:
            if keys[pygame.K_RIGHT] and Action.RIGHT in user_permissible_actions:
                action_taken = Action.RIGHT
            elif keys[pygame.K_LEFT] and Action.LEFT in user_permissible_actions:
                action_taken = Action.LEFT
            elif keys[pygame.K_UP] and Action.TOP in user_permissible_actions:
                action_taken = Action.TOP
            elif keys[pygame.K_DOWN] and Action.DOWN in user_permissible_actions:
                action_taken = Action.DOWN
    else:
        action_taken = None
    return action_taken

"""This function is used to draw the body of the snake, display score, display food points
for a manual game of snake"""

def runRandomGame(play=True, scalingFactor = 9):  # Scaling the size of the grid):
    g = Game.Game()  # Instantiating an object of class Game
    width = scalingFactor * Constants.gridSize
    height = scalingFactor * Constants.gridSize
    pos_score_x = int(math.floor(width / (Constants.numberOfSnakes + 1)))  # For displaying score
    pos_score_y = int(math.floor(height / 20))
    font_size = int(math.floor(height / 40))
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    colors = np.random.randint(0, 256, size=[Constants.numberOfSnakes, 3])
    if play: # user interacts with the agents
        colors[0] = black # player's snake is always black
    crashed = False
    episodeRunning = True
    win = pygame.display.set_mode((scalingFactor * Constants.gridSize, scalingFactor * Constants.gridSize))  # Game Window
    screen = pygame.Surface((Constants.gridSize+1, Constants.gridSize+1))  # Grid Screen
    pygame.display.set_caption("Snake Game")
    clock = pygame.time.Clock()

    while not crashed and episodeRunning:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
        screen.fill(white)
        #draw the walls
        pygame.draw.lines(screen, black, True, [(0,0), (0,Constants.gridSize), (Constants.gridSize, Constants.gridSize), (Constants.gridSize, 0)])
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

        actionsList = manual_action_list(g, event)
        """
        actionsList = rl_agent(g)
        Can also add an if condition here to have one player and one agent
        and then append the two lists obtained, to pass it to move method
        """
        _, episodeRunning = g.move(actionsList)
        win.blit(pygame.transform.scale(screen, win.get_rect().size), (0, 0)) # Transforms the screen window into the win window
        for idx in range(Constants.numberOfSnakes):
            draw_text(win, "Snake" + str(idx) + "  " + str(g.snakes[idx].score), font_size, pos_score_x * (idx + 1), pos_score_y,black) #Displaying score
        pygame.display.update()
        clock.tick(10)  # (FPS)means that for every second at most 10 frames should pass.
    pygame.quit()



def displayGame(game, win, screen, colors, scalingFactor = 9):
    width = scalingFactor * Constants.gridSize
    height = scalingFactor * Constants.gridSize
    pos_score_x = int(math.floor(width / (Constants.numberOfSnakes + 1)))  # For displaying score
    pos_score_y = int(math.floor(height / 20))
    font_size = int(math.floor(height / 40))
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)

    clock = pygame.time.Clock()
    screen.fill(white)
    #draw the walls
    pygame.draw.lines(screen, black, True, [(0,0), (0,Constants.gridSize), (Constants.gridSize, Constants.gridSize), (Constants.gridSize, 0)])
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

    win.blit(pygame.transform.scale(screen, win.get_rect().size), (0, 0)) # Transforms the screen window into the win window
    for idx in range(len(game.snakes)):
        draw_text(win, "Snake" + str(idx) + "  " + str(game.snakes[idx].score), font_size, pos_score_x * (idx + 1), pos_score_y,black) #Displaying score
    pygame.display.update()
    clock.tick(10)  # (FPS)means that for every second at most 10 frames should pass.
    return

if __name__ == '__main__':
    runRandomGame()
