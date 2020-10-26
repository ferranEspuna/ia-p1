# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"

        doneCorners = tuple((self.startingPosition == corner for corner in self.corners)) # En los estados anotamos las esquinas por las que hemos empezado
        firstState = (self.startingPosition, doneCorners)
        self.startState = firstState
        self.costFn = lambda x: 1

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        return sum(state[1]) == 4 #Si pacman ha comido las 4 unidades de comida, hemos ganado

    def getSuccessors(self, state):

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            x,y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if not hitsWall:

                nextCorners = tuple((before or ((nextx, nexty) == corner) for corner, before in zip(self.corners, state[1]))) #TODO comentar esta vaina

                nextState = ((nextx, nexty), nextCorners)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem): #TODO comentar esta vaina
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"




    unvisitedCorners = []
    for i in range(4):
        if not state[1][i]:
            unvisitedCorners.append(corners[i])


    unvisitedCornersNumber = len(unvisitedCorners)  # Número de esquinas que no hemos visitado

    x = state[0][0]
    y = state[0][1]
    w = corners[3][0] - 1
    h = corners[3][1] - 1

    if unvisitedCornersNumber == 0:
        return 0

    elif unvisitedCornersNumber == 1:
        return abs(unvisitedCorners[0][0] - x) + abs(unvisitedCorners[0][1] - y)

    elif unvisitedCornersNumber == 2:
        return min([abs(x - x0) + abs(y - y0) for x0, y0 in unvisitedCorners]) + abs(
            unvisitedCorners[0][0] - unvisitedCorners[1][0]) + abs(unvisitedCorners[0][1] - unvisitedCorners[1][1])

    elif unvisitedCornersNumber == 3:

        missingCornerIndex = state[1].index(True)
        missingCorner = (corners[missingCornerIndex])

        a = abs(x - missingCorner[0])
        b = abs(y - missingCorner[1])
        aprime = w - a
        bprime = h - b

        return min([b + aprime + w + h,
                    a + bprime + w + h,
                    aprime + bprime + h + 2 * w,
                    aprime + bprime + w + 2 * h])

    else:

        return max(w, h) + 2 * min(w, h) + min([abs(x - x0) + abs(y - y0) for x0, y0 in corners])



class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

class BFSFoodSearchAgent(SearchAgent):#Para poder comparar la eficiencia de a* con la de bfs en el problema de la comida
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.bfs(prob)
        self.searchType = FoodSearchProblem

def foodHeuristic_closest(state, problem): #Como mínimo el coste será ir a la comida más cercana, y luego una unidad extra por cada posición con comida distinta a la más cercana

    position, foodGrid = state

    if foodGrid.count() == 0: return 0
    return closestDot(position, problem, foodGrid)[1] + foodGrid.count() - 1
    return foodGrid.count()

def foodHeuristic_farthest(state, problem): #Como mínimo el coste será ir a la posición más lejana con comida

    position, foodGrid = state

    if foodGrid.count() == 0: return 0
    return farthestDot(position, problem, foodGrid)[1]

def foodHeuristic_farthestApart(state, problem):

    position, foodGrid = state

    farthest1 = position
    farthest2 = position
    farthestdistance = 0

    for x in range(foodGrid.width):
        for y in range(foodGrid.height):
            if foodGrid[x][y]:#Para cada posición sin comida:

                for (x2, y2), distance in reversed(findAdjacencyCheck((x, y), problem)): #Empezamos mirando las posiciones del segundo desde la más lejana al primero
                    if foodGrid[x2][y2]: #Esta es la posición con comida más alejada de (x, y)

                        if distance > farthestdistance:#Si la distancia entre los nodos es más grande que las que hemos mirado hasta ahora, lo actualizamos

                            farthest1 = (x, y)
                            farthest2 = (x2, y2)
                            farthestdistance = distance

    addition = [dist for pos, dist in findAdjacencyCheck(position, problem) if (pos == farthest1 or pos == farthest2)]

    return min(addition) + farthestdistance

def foodHeuristic_mst(state, problem): #minimum spanning tree

    position, foodGrid = state

    if foodGrid.count() == 0: return 0
    cd = closestDot(position, problem, foodGrid)[1]
    minst = mstCheck(problem, foodGrid)

    return cd + minst

"""
Verbose nos permite ir imprimendo por pantalla las heurísticas más bajas obtenidas,
lo cual nos da una idea del "progreso" de la ejecución (cuando llega a 0, hemos acabado.

Nótese que lo cerca que estamos de la solución no depende linealmente de esta heurística:
como nos centramos en los nodos con suma heurística + camino más baja, cuanto más baje la heurística
tendremos exponencialmente más nodos por explorar.

Heuristic va a ser una de las funciones anteriores marcadas como foodHeuristic_XXX

"""

def foodHeuristic(state, problem, verbose = True, heuristic = foodHeuristic_mst):

    if verbose and not "heuristicRecord" in problem.heuristicInfo:
        problem.heuristicInfo["heuristicRecord"] = 999999999


    h = heuristic(state, problem)


    if verbose and h < problem.heuristicInfo["heuristicRecord"]:
        problem.heuristicInfo["heuristicRecord"] = h
        print(h)

    return h




class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"

        afsp = AnyFoodSearchProblem(gameState)
        return search.bfs(afsp)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE


    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state
        return self.food[x][y]

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))

def findPathToClosestDot(position, food, problem):
    afsp = AnyFoodSearchProblem2(position, food, problem)
    return search.bfs(afsp)

class AnyFoodSearchProblem2(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, position, food, problem):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = food

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = problem.walls
        self.startState = position
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE


    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state
        return self.food[x][y]


#Esta clase crea un problemaen el que ninguna posición es el objetivo, para que dada una posición inicial se encuentren las distancias a las otras posiciones
class allPositionSearchProblem(PositionSearchProblem):

    def __init__(self, position, problem):

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = problem.walls
        self.startState = position
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE


    def isGoalState(self, state):
        return False

def findAdjacency(position, problem): #hace bfs (sin parar cuando llega a ningún nodo) para encontrar las distancias más cortas a cada posición

    apsp = allPositionSearchProblem(position, problem)
    return search.bfs(apsp)

def findAdjacencyCheck(position, problem): #No calcula cada vez el coste de llegar a cada posición, solo una vez por nodo

    if not "adjacency" in problem.heuristicInfo:
        problem.heuristicInfo["adjacency"] = dict()

    if not position in problem.heuristicInfo["adjacency"]:
        problem.heuristicInfo["adjacency"][position] = findAdjacency(position, problem)

    return problem.heuristicInfo["adjacency"][position]

def closestDot(position, problem, food): #Encuentra la posición más cercana con comida, con su coste.

    for point in findAdjacencyCheck(position, problem):
        if food[point[0][0]][point[0][1]]:
            return point


def farthestDot(position, problem, food): #Encuentra la posición más lejana con comida, con su coste.

    for point in reversed(findAdjacencyCheck(position, problem)):
        if food[point[0][0]][point[0][1]]:
            return point


"""
Los métodos a continuación tienen por objetivo calcular un minimum spanning tree de los puntos de comida restantes
en un mapa concreto, usando los puntos como nodos y su distancia real entre ellos (calculada con un bfs) como pesos de las aristas.
Esto se usa en la heurística foodHeuristic_mst
"""


def update(listn):

    if len(listn) == 0: return 9999999999
    return listn[0][2]

def minIndex(listofnumbers):

    minimum = listofnumbers[0]
    index = 0
    for idx, number in enumerate(listofnumbers):
        if number < minimum:
            index = idx
            minimum = number

    return index, minimum

def gridToHashable(boolgrid):

    return str(boolgrid)

def join(pos1, pos2, connectionsDict):

    winner = min(connectionsDict[pos1], connectionsDict[pos2])
    loser = max(connectionsDict[pos1], connectionsDict[pos2])
    for key in connectionsDict.keys():
        if connectionsDict[key] == loser:
            connectionsDict[key] = winner


def isdone(connectionsDict):

    l = list(connectionsDict.values())
    first = l[0]
    for value in l:
        if value != first:
            return False

    return True

def areJoined(pos1, pos2, connectionsDict):
    return connectionsDict[pos1] == connectionsDict[pos2]

def mst(problem, food):#Esto es una implementación del algoritmo de Kruskal, optimizada para funcionar con listas de adyacencia ordenadas


    edgesLists = [[((x1, y1), (x2, y2), cost) for (x2, y2), cost in findAdjacencyCheck((x1, y1), problem) if food[x2][y2]] for x1 in range(food.width) for y1 in range(food.height) if food[x1][y1]]
    mincostList = [update(listn) for listn in edgesLists]
    connectionsDict = {item[0][0]: number for number, item in enumerate(edgesLists)}
    currentCost = 0

    while(not isdone(connectionsDict)):

        idx, minimum = minIndex(mincostList)

        point1, point2, cost = edgesLists[idx].pop(0)
        mincostList[idx] = update(edgesLists[idx])

        if not areJoined(point1, point2, connectionsDict):

            join(point1, point2, connectionsDict)
            currentCost += cost

    return currentCost


def mstCheck(problem, food):#Si ya se ha calculado el mst para una colección de puntos (foodGrid), no hace falta volver a calcularlo

    if not "mst" in problem.heuristicInfo:
        problem.heuristicInfo["mst"] = dict()

    n = gridToHashable(food)

    if n in problem.heuristicInfo["mst"]: return problem.heuristicInfo["mst"][n]

    minst = mst(problem, food)
    problem.heuristicInfo["mst"][n] = minst
    return minst







