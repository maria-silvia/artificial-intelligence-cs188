# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def myGenericSearch(problem: SearchProblem, fronteira):
    """
    Método genérico sugerido pelo enunciado do projeto.
    Implementa todas as buscas de acordo com a fronteira passada.
    Parametro fronteira deve conter a estrategia de fila de como cada busca 
    escolhe o proximo nodo pra expansao 
    """
    states_visitados = set()

    fronteira.push((problem.getStartState(), []))
    while not fronteira.isEmpty():
        [state, caminho] = fronteira.pop()

        if problem.isGoalState(state):
            return caminho

        if state not in states_visitados:
            states_visitados.add(state)

            for s_state, s_action, _cost in problem.getSuccessors(state):
                if s_state not in states_visitados:
                    caminhoAteSucessor = caminho + [s_action]
                    fronteira.push((s_state, caminhoAteSucessor))

    return []  # sem solucao


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    fronteira = util.Stack()
    return myGenericSearch(problem, fronteira)

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    fronteira = util.Queue()
    return myGenericSearch(problem, fronteira)
    
def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    def custoDoCaminho(successor):
        _, actions = successor
        return problem.getCostOfActions(actions)
    
    fronteira = util.PriorityQueueWithFunction(custoDoCaminho)
    return myGenericSearch(problem, fronteira)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    def custoDoCaminhoMaisHeuristica(successor): 
        state, actions = successor
        return problem.getCostOfActions(actions) + heuristic(state, problem)
    
    fronteira = util.PriorityQueueWithFunction(custoDoCaminhoMaisHeuristica)
    return myGenericSearch(problem, fronteira)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
