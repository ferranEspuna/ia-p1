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

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def justAdd(element, structure, heuristic=None, problem=None):
    structure.push(element)

def addPriority(element, structure, heuristic, problem): #Usaremos esta función para A* y para UCS, pues estos algoritmos usan una cola de prioridades a las cuales hay que pasar la prioridad.
    structure.push(element, element[2] + heuristic(element[0], problem))


"""
A continuación veremos dos implementaciones generales de algoritmos de búsqueda.
Estas reciben por parámetro una estructura de datos (por ejemplo, una pila) que usarán a modo de frontera.
Dependiendo de la estructura que pasemos, el algoritmo implementado será uno u otro.
"""
def explore(problem, structure, addFunction=justAdd, heuristic=nullHeuristic):

    start = problem.getStartState()
    frontera = structure() # Aquí tendremos nodos de la forma (state, moves), donde moves es una lista con los movimientos que nos llevan a ese estado
    visited = set()  # Conjunto de los nodos (estados) que ya hemos visitado (expandido)
    addFunction((start, [], 0), frontera, heuristic, problem) #La frontera empieza siendo solo el estado inicial

    while not frontera.isEmpty():

        currentState, currentMoves, currentCost = frontera.pop()  # Extraemos el siguiente estado a exprlorar de la estructura, junto a los movimientos que nos han llevado a este

        if problem.isGoalState(currentState):  # Si el sucesor es el objetivo, devolvemos los movimientos que nos han llevado a él
            return currentMoves

        if not currentState in visited:  # Si ya hemos visitado el nodo, hay que ignorarlo

            visited.add(currentState)  # En caso contrario, lo añadimos a visited y lo expandimos

            for successor, action, extraCost in problem.getSuccessors(currentState):  #Para cada sucesor:

                actions = currentMoves + [action]  #Las acciones para llegar al sucesor son las que nos han llevado al actual más la que nos lleva del actual al sucesor
                addFunction((successor, actions, currentCost + extraCost), frontera, heuristic, problem) #frontera.push((successor, actions))  # Lo añadimos a la cola, con el movimiento extra que le corresponda

    return []

def explore2(problem, structure, addFunction=justAdd, heuristic=nullHeuristic): #Mismo concepto, pero ahorramos memoria no guardando todos los caminos parciales

    start = problem.getStartState()
    frontera = structure()
    visited = dict() #En visited guardaremos además el padre y la acción que nos han llevado al hijo
    sortedlist = []
    addFunction((start, "noParent", 0,"noAction"), frontera, heuristic, problem) #En la pila ahora no pasamos caminos parciales, solo padres y acciones

    while not frontera.isEmpty():

        currentState, currentParent, currentCost, previousAction = frontera.pop()

        if problem.isGoalState(currentState): #Aquí hay que reconstruir el camino, pasando por todos los nodos

            moves = [previousAction]
            state = currentParent

            while state in visited:
                moves.append(visited[state][1])
                state = visited[state][0]

            moves.reverse()
            return moves[1:]

        if not currentState in visited:

            visited[currentState] = (currentParent, previousAction, currentCost)
            sortedlist.append((currentState, currentCost)) #Vamos añadiendo los nodos a una lista, que los contiene junto con el coste de llegar a ellos

            for successor, action, extraCost in problem.getSuccessors(currentState):

                addFunction((successor, currentState, currentCost + extraCost, action), frontera, heuristic, problem)

    return sortedlist #Si ningún nodo era goalstate, devolvemos todos los nodos, ordenados por coste




def depthFirstSearch(problem):
    return explore2(problem, util.Stack) #Implementamos el algoritmo con una pila

def breadthFirstSearch(problem):
    return explore2(problem, util.Queue)  # Implementamos el algoritmo con una cola

def uniformCostSearch(problem): #Implementamos el algoritmo con una cola de prioridades
    return explore2(problem, util.PriorityQueue, addPriority, nullHeuristic)

def aStarSearch(problem, heuristic=nullHeuristic): #Implementamos el algoritmo con una cola de prioridades, pasando si nos la dan una heurística
    return explore2(problem, util.PriorityQueue, addPriority, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
