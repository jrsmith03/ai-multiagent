# --------------
# multiAgents.py
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
import math

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if (manhattanDistance(newGhostStates[0].getPosition(), newPos) < 2) : 
            print ("GHOST")
            return -10000
        food = currentGameState.getFood().asList()
        closest_food = 10000000
        for foo in food : 
            dist = manhattanDistance(newPos, foo)
            closest_food = min(dist, closest_food)
        
        return -closest_food 

        

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # minvalue function called by the ghosts
        def min_ghost(gameState: GameState, depth, ghostID) : 
            if (gameState.isWin() or gameState.isLose() or depth == 0) : 
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(ghostID)

            value = math.inf
            for action in legalActions : 
                if ghostID == gameState.getNumAgents() - 1:
                    value = min(value, max_pacman(gameState.generateSuccessor(ghostID, action), depth - 1))
                else:
                    # Otherwise, move to the next ghost
                    value = min(value, min_ghost(gameState.generateSuccessor(ghostID, action), depth, ghostID + 1))
            return value

        # maxvalue function called by pacman
        def max_pacman(gameState: GameState, depth) :
            if (gameState.isWin() or gameState.isLose() or depth == 0) : 
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)

            value = -math.inf
            for action in legalActions : 
                value = max(value, min_ghost(gameState.generateSuccessor(0, action), depth, 1))
            return value

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = -math.inf
        for action in legalActions :
            value = min_ghost(gameState.generateSuccessor(0, action), self.depth, 1)
            if (value > bestValue) : 
                bestValue = value
                bestAction = action
        return bestAction
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
                # minvalue function called by the ghosts
        def min_ghost(gameState: GameState, depth, ghostID, alpha, beta) : 
            if (gameState.isWin() or gameState.isLose() or depth == 0) : 
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(ghostID)

            value = math.inf
            for action in legalActions : 
                if ghostID == gameState.getNumAgents() - 1:
                    value = min(value, max_pacman(gameState.generateSuccessor(ghostID, action), depth - 1, alpha, beta))
                else: 
                    # Otherwise, move to the next ghost
                    value = min(value, min_ghost(gameState.generateSuccessor(ghostID, action), depth, ghostID + 1, alpha, beta))
                if value < alpha :
                    return value
                beta = min(beta, value)
            return value

        # maxvalue function called by pacman
        def max_pacman(gameState: GameState, depth, alpha, beta) :
            if (gameState.isWin() or gameState.isLose() or depth == 0) : 
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)

            value = -math.inf
            for action in legalActions : 
                value = max(value, min_ghost(gameState.generateSuccessor(0, action), depth, 1, alpha, beta))
                if value > beta :
                    break
                alpha = max(alpha, value)
            return value

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = -math.inf

        alpha = -math.inf
        beta = math.inf

        for action in legalActions :
            value = min_ghost(gameState.generateSuccessor(0, action), self.depth, 1, alpha, beta)
            if (value > bestValue) : 
                bestValue = value
                bestAction = action
            if value > beta :
                break
            alpha = max(alpha, bestValue)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # minvalue function called by the ghosts
        def expected_ghost(gameState: GameState, depth, ghostID) : 
            if (gameState.isWin() or gameState.isLose() or depth == 0) : 
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(ghostID)
            expectedValue = 0        
            for action in legalActions : 
                if ghostID == gameState.getNumAgents() - 1:
                    # only depth - 1 if the last ghost
                    value = max_pacman(gameState.generateSuccessor(ghostID, action), depth - 1)
                    expectedValue += value
                else:
                    # Otherwise, move to the next ghost
                    value = expected_ghost(gameState.generateSuccessor(ghostID, action), depth, ghostID + 1)
                    expectedValue += value
            return expectedValue / len(legalActions)
# `
            
        # maxvalue function called by pacman
        def max_pacman(gameState: GameState, depth) :
            if (gameState.isWin() or gameState.isLose() or depth == 0) : 
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)
            bestValue = -math.inf
            for action in legalActions : 
                value = expected_ghost(gameState.generateSuccessor(0, action), depth, 1)
                bestValue = max(bestValue, value)
            return bestValue

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = -math.inf
        for action in legalActions :
            value = expected_ghost(gameState.generateSuccessor(0, action), self.depth, 1)
            if (value > bestValue) : 
                bestValue = value
                bestAction = action
        return bestAction



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    pelletList = currentGameState.getCapsules()
    
    minFoodDist = math.inf
    minGhostDist = math.inf
    minPelletDist = math.inf
    totalPelletDist = 0

    totalFoodDist = 0
    for food in foodList.asList() :
        minFoodDist = min(minFoodDist, manhattanDistance(newPos, food))
        totalFoodDist += manhattanDistance(newPos, food)
    for ghost in ghostStates :
        minGhostDist = min(minGhostDist, manhattanDistance(newPos, ghost.getPosition()))
    for pellet in pelletList :
        minPelletDist = min(minPelletDist, manhattanDistance(newPos, pellet))
        totalPelletDist += minPelletDist
    
    heuristic_score = 0

    # Penalties
    # if minGhostDist > 3 and minFoodDist < 20 :
    #     heuristic_score -= 5*minFoodDist
    if len(pelletList) > 0 :
        heuristic_score -= 200 / (minGhostDist + 100)
    if minGhostDist < 3:
        # incentivize being closer to all food and further from ghost
        heuristic_score -= 1e5 
    else:
        # incentivize being further from ghost and eating more food
        # heuristic_score += 100 - totalFoodDist + minGhostDist - len(foodList.asList())*10
        heuristic_score += 100 - totalFoodDist + minGhostDist - len(foodList.asList())*10
    
    return heuristic_score 

# Abbreviation
better = betterEvaluationFunction
