# multiAgents.py
# --------------
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

# Trabalho 2 de IA 
# MARIA SILVIA RIBEIRO RUY
# grr 20182587

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from pacman import SCARED_TIME

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
        # + action faz ganhar o jogo
        if successorGameState.isWin():
            return float("inf")
        # - action faz perder o jogo
        if successorGameState.isLose():
            return float("-inf")

        # + action assusta os fantasmas
        if newScaredTimes[0] == SCARED_TIME:
            return float("inf")
         
        # - action aproxima muito o pacman de algum fantasma nao assustado
        # ou assustado mas por pouco tempo
        assustados = newScaredTimes[0] != 0
        if not assustados or newScaredTimes[0] < 3:
            for ghost in newGhostStates:
                if manhattanDistance(ghost.getPosition(), newPos) < 3:
                    return float("-inf")
            
        # retorna score proporcional a comida mais proxima
        distanciaComidas = [manhattanDistance(food, newPos) for food in newFood.asList()]
        temComidaNaPosicao = currentGameState.getNumFood() > successorGameState.getNumFood()
            
        score = successorGameState.getScore()
        score -= 5 * min(distanciaComidas)
        score += 100 if temComidaNaPosicao else 0
        
        return score

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

        # *** FAZ DFS EXAUTIVVA ***

        bestValue = float("-inf")
        bestAction = None 

        for action in gameState.getLegalActions(0):
            nodoFilhoState = gameState.generateSuccessor(0, action)
            nodoFilhoStateValue = self.value(nodoFilhoState, 1, depth=0)

            if nodoFilhoStateValue > bestValue:
                bestValue = nodoFilhoStateValue
                bestAction = action
        
        return bestAction

    def value(self, gameState, agentIndex, depth):
        """
            Dispatcher 
        """
        isPacman = agentIndex == 0
        if isPacman:
            depth += 1

        # caso base, folha
        isTerminalState = gameState.isWin() \
            or gameState.isLose() \
            or depth == self.depth \

        if isTerminalState:
            return self.evaluationFunction(gameState)
        
        if isPacman:
            return self.maxValue(gameState, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, depth):
        v = float("-inf")
        for action in gameState.getLegalActions(0):
            sucessor = gameState.generateSuccessor(0, action)
            v_sucessor = self.value(sucessor, 1, depth)
            
            v = max(v, v_sucessor)
        return v

    def nextAgent(self, currentAgent, gameState):
        return (currentAgent + 1) % gameState.getNumAgents()
    
    def minValue(self, gameState, agentIndex, depth):
        v = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            sucessor = gameState.generateSuccessor(agentIndex, action)
            nextAgent = self.nextAgent(agentIndex, gameState)
            v_sucessor = self.value(sucessor, nextAgent, depth)

            v = min(v, v_sucessor)
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # *** FAZ DFS EXAUTIVVA ***
        bestValue = float("-inf") # a melhor opcao para o pacman
        bestAction = None 
        beta = float("inf") # a melhor opcao para os fantasmas

        for action in gameState.getLegalActions(0):
            nodoFilhoState = gameState.generateSuccessor(0, action)
            nodoFilhoStateValue = self.value(
                nodoFilhoState, 
                agentIndex=1, 
                depth=0, 
                alpha=bestValue, 
                beta=beta
            )

            if nodoFilhoStateValue > bestValue:
                bestValue = nodoFilhoStateValue
                bestAction = action

        return bestAction
    
    def value(self, gameState, agentIndex, depth, alpha, beta):
        """
            Dispatcher 
        """
        isPacman = agentIndex == 0
        if isPacman:
            depth += 1

        # caso base, folha
        isTerminalState = gameState.isWin() \
            or gameState.isLose() \
            or depth == self.depth \

        if isTerminalState:
            return self.evaluationFunction(gameState)
        
        if isPacman:
            return self.maxValue(gameState, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)


    def nextAgent(self, currentAgent, gameState):
        return (currentAgent + 1) % gameState.getNumAgents()
    

    def maxValue(self, gameState, depth, alpha, beta):
        v = float("-inf")
        for action in gameState.getLegalActions(0):
            sucessor = gameState.generateSuccessor(0, action)
            v_sucessor = self.value(sucessor, 1, depth, alpha, beta)

            v = max(v, v_sucessor)
            if v > beta:
                # nao precisa continuar explorando, mesmo se tiver valor maior pela frente
                # o fantasma nao vai escolher este ramo pois outro ramo ja oferece valor melhor (beta)
                return v
            alpha = max(alpha, v)    
            
        return v

    
    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        v = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            sucessor = gameState.generateSuccessor(agentIndex, action)
            nextAgent = self.nextAgent(agentIndex, gameState)
            v_sucessor = self.value(sucessor, nextAgent, depth, alpha, beta)

            v = min(v, v_sucessor)
            if v < alpha:
                # nao precisa continuar explorando, mesmo se tiver valor menor pela frente
                # o pacman nao vai escolher este ramo pois outro ramo ja oferece valor melhor (alpha)
                return v 
            beta = min(beta, v) 
        return v

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
        # *** FAZ DFS EXAUTIVVA ***
        bestValue = float("-inf")
        bestAction = None 

        for action in gameState.getLegalActions(0):
            nodoFilhoState = gameState.generateSuccessor(0, action)
            nodoFilhoStateValue = self.value(nodoFilhoState, 1, depth=0)

            if nodoFilhoStateValue > bestValue:
                bestValue = nodoFilhoStateValue
                bestAction = action
        return bestAction
    
    def value(self, gameState, agentIndex, depth):
        """
            Dispatcher 
        """
        isPacman = agentIndex == 0
        if isPacman:
            depth += 1

        # caso base, folha
        isTerminalState = gameState.isWin() \
            or gameState.isLose() \
            or depth == self.depth \

        if isTerminalState:
            return self.evaluationFunction(gameState)
        
        if isPacman:
            return self.maxValue(gameState, depth)
        else:
            return self.expValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, depth):
        v = float("-inf")
        for action in gameState.getLegalActions(0):
            sucessor = gameState.generateSuccessor(0, action)
            v_sucessor = self.value(sucessor, 1, depth)
            
            v = max(v, v_sucessor)
        return v

    def nextAgent(self, currentAgent, gameState):
        return (currentAgent + 1) % gameState.getNumAgents()
    
    # unica diferenca pro MinimaxAgent: 
    def expValue(self, gameState, agentIndex, depth):
        values = 0
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            sucessor = gameState.generateSuccessor(agentIndex, action)
            nextAgent = self.nextAgent(agentIndex, gameState)
            v_sucessor = self.value(sucessor, nextAgent, depth)

            values += v_sucessor
        
        return values / len(legalActions)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Estima -inf para quando fantasmas estão muito pertos, estima alto valor 
     para quando da para comer fantasmas ou quando ha comida perto
    """
    "*** YOUR CODE HERE ***"

    # score defaiult
    score = scoreEvaluationFunction(currentGameState)  
    newPos = currentGameState.getPacmanPosition()
    
    # regula pra fantasmas
    newGhostStates = currentGameState.getGhostStates()
    for ghost in newGhostStates:
        distancia = manhattanDistance(ghost.getPosition(), newPos) 
        assustado = ghost.scaredTimer > 0;
        if assustado:
            score -= 10 / distancia
        if distancia < 1:
            return float("-inf")

    # regula pra comidas
    newFood = currentGameState.getFood()
    distanciaComidas = [manhattanDistance(food, newPos) for food in newFood.asList()]
    score -= 2 * min(distanciaComidas, default=0)
    score -= max(distanciaComidas, default=0)
    score -= 8 * currentGameState.getNumFood()
    
    # regula pra pellets
    capsulas = currentGameState.getCapsules()
    distanciaCapsulas = [manhattanDistance(c, newPos) for c in capsulas]
    score -= 3 * min(distanciaCapsulas, default=0) 

    return score

# Abbreviation
better = betterEvaluationFunction
