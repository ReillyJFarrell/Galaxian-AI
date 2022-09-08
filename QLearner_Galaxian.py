import random
import sys
from ale_py import ALEInterface
import numpy as np
from pynput.keyboard import Controller

# ALL PERFORMANCE EVALUATIONS WERE DONE WITH DISPLAY ON AT 0MS DELAY, RESULTS WILL VARY IF THIS ENVIRONMENT IS
# NOT THE SAME. TO SPEED UP THE ROM, PRESS 'S' 4 TIMES ON STARTUP, See Line 85 for more information

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} rom_file episodes")
    sys.exit()

# initialization
ale = ALEInterface()
np.set_printoptions(threshold=np.inf)
ale.setInt("random_seed", 123)

# Qlearning Setup
EXPLORE_PROB = 0.2
LEARNING_RATE_MOVING = 0.35
LEARNING_RATE_DISTANCE = 0.15
SAME_SCORE_MOVE_REWARD = -1000
SAME_SCORE_DISTANCE_REWARD = -300
EDGE_BOUNDS = 10
EDGE_REWARD = -10000

LOWER_BOUND_BULLETS = 170
UPPER_BOUND_BULLETS = 185
LEFT_BOUND_BULLETS = 15
RIGHT_BOUND_BULLETS = 145

# How much we reward for a better score multiplyer
SCORE_INCR_MULT = 1
DISCOUNT_FACTOR_MOVE = 0.8
DISCOUNT_FACTOR_DISTANCE = 0.8
# Arbitrary value for deciding which way to move
MOVE_VAL = 35

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
if USE_SDL:
    ale.setBool("sound", True)
    ale.setBool("display_screen", True)

# Load the ROM file
rom_file = sys.argv[1]
runEpisodes = int(sys.argv[2])
ale.loadROM(rom_file)

# Actions are:
# 0 nothing
# 1 fire
# 2 up
# 3 right
# 4 left
# 5 down
# 11 right fire
# 12 left fire
# We only care about 11 and 12

# 216 is grayscale val of bullet


# Q values of moving left or right, initialized to 0 each
leftRightQtableActionValues = [0, 0]
# Possible moves
leftRightQtableActions = [-MOVE_VAL, MOVE_VAL]

# Q values for how far the ship should move left if chosen, initialized to 0 each
distanceQtableDistanceL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Possible distances to travel left
distanceQtableDistanceValuesL = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# Q values for how far the ship should move right if chosen, initialized to 0 each
distanceQtableDistanceR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Possible distances to travel right
distanceQtableDistanceValuesR = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Stores the scores of each episode
scores = []

# Used to ensurer game is run in the environment it was developed in, set auto to false for manual speed adjustment
# If speed is not automatically set to 0MS, increase KEYBOARD_SPEED_SETTER to 30 or manually press 's'
AUTO = True
# How many times to loop pressing the speedup key
KEYBOARD_SPEED_SETTER = 6
keyboard = Controller()
key = "s"






for episode in range(runEpisodes):
    # Previous score of the game
    # (Used with current score to see how much reward we have gained since performing an action)
    prevScore = 0
    # Current score of the game
    currentScore = 0

    while not ale.game_over():
        if KEYBOARD_SPEED_SETTER != 0 and AUTO:
            keyboard.press(key)
            keyboard.release(key)
            KEYBOARD_SPEED_SETTER -= 1
        # Determine if we use epsilon greedy exploration for direction we move
        chance = random.randint(1, 100)
        if chance <= EXPLORE_PROB * 100:
            index = random.randint(0, len(leftRightQtableActionValues) - 1)
            command = leftRightQtableActions[index]
        else:
            command = leftRightQtableActions[leftRightQtableActionValues.index(max(leftRightQtableActionValues))]

        if command < 0:
            action = 12
        else:
            action = 11

        if action == 11:
            actionChoice = distanceQtableDistanceValuesL
            distQTable = distanceQtableDistanceL

        else:
            actionChoice = distanceQtableDistanceValuesR
            distQTable = distanceQtableDistanceR
        # Determine if we use epsilon greedy exploration for distance we move
        chance = random.randint(1, 100)
        if chance <= EXPLORE_PROB * 100:
            val = random.randint(0, len(actionChoice) - 1)
            hold = actionChoice[val]
            actionCount = hold

        else:
            hold = actionChoice[distQTable.index(max(distQTable))]
            actionCount = hold
        prevScore = currentScore
        # While loop to continue acting until our duration is depleted
        while actionCount != 0:
            # Get screen state
            s = ale.getScreen()
            # Crop based on bounds specified
            crop_screen = s[LOWER_BOUND_BULLETS:UPPER_BOUND_BULLETS, LEFT_BOUND_BULLETS:RIGHT_BOUND_BULLETS]
            total = 0
            # Sum the pixels to determine if there are any other objects in the cropped screen besides the ship
            for pixels in crop_screen:
                total += np.sum(pixels)
                # We know at least one object is in the cropped screen so break to avoid counting all
                if total > 2072:
                    break
            # Another possible safe state
            if total == 1260:
                # State where only the ship is inside the cropped screen, which deemed safe
                total = 2072
            # If there is another object inside the cropped screen that can kill the agent
            if total != 2072:
                # Do not move
                reward = ale.act(1)
                # Necessary for maintaining accurate record of score
                currentScore += reward
                continue
            # If there are no objects that are dangerous in the cropped screen, we continue to act as intended
            reward = ale.act(action)
            currentScore += reward
            actionCount -= 1

        # After the agent has finished acting it checks if it's too close to an edge, specified by a constant
        screen = ale.getScreen()
        cropped = screen[178:179, 15:145]
        leftEdge = cropped[0][1:EDGE_BOUNDS + 1]
        end = len(cropped[0]) - 1
        rightEdge = cropped[0][end - EDGE_BOUNDS:end]
        both = sum(leftEdge) + sum(rightEdge)
        edgePenalty = 0
        # Ship consists of a few 14 pixel values, so if the edge bound is divisible by 14,
        # we know the ship must be near one of them
        if both % 14 == 0 and both > 0:
            # Apply penalty for being at the edge of the screen
            edgePenalty = EDGE_REWARD

        # Determine reward if ship has not gained any additional points
        if prevScore == currentScore:
            # Action Q table update
            old = leftRightQtableActionValues[leftRightQtableActions.index(command)]
            leftRightQtableActionValues[leftRightQtableActions.index(command)] \
                = old + LEARNING_RATE_MOVING * (SAME_SCORE_MOVE_REWARD + edgePenalty + DISCOUNT_FACTOR_MOVE *
                                                max(leftRightQtableActions) - old)
            # Distance Q table update
            old = distQTable[actionChoice.index(hold)]
            distQTable[actionChoice.index(hold)] \
                = old + LEARNING_RATE_DISTANCE * (SAME_SCORE_DISTANCE_REWARD + DISCOUNT_FACTOR_DISTANCE *
                                                  max(actionChoice) - old)
        else:
            # Action Q table update
            old = leftRightQtableActionValues[leftRightQtableActions.index(command)]
            leftRightQtableActionValues[leftRightQtableActions.index(command)] \
                = old + LEARNING_RATE_MOVING * (edgePenalty + SCORE_INCR_MULT * (currentScore - prevScore) +
                                                DISCOUNT_FACTOR_MOVE *
                                                max(leftRightQtableActions) - old)
            # Distance Q table update
            old = distQTable[actionChoice.index(hold)]
            distQTable[actionChoice.index(hold)] \
                = old + LEARNING_RATE_DISTANCE * (SCORE_INCR_MULT * (currentScore - prevScore) +
                                                  DISCOUNT_FACTOR_DISTANCE *
                                                  max(actionChoice) - old)
        # Update our scores
        prevScore = currentScore
    # Episode has concluded, print and store results
    print("Episode {} ended with score: {}".format(episode, prevScore))
    scores.append(prevScore)
    ale.reset_game()
# All episodes have concluded, print data
print("Average score of {} episodes is: {}".format(runEpisodes, np.average(scores)))
print("Max score was: {}".format(np.max(scores)))