from captureAgents import CaptureAgent
from baselineTeam import ReflexCaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Actions

DEBUG = False
CHEAT = False

class ApproximateQAgent(CaptureAgent):

    def __init__( self, index ):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.numTraining = 0
        self.episodesSoFar = 0
        self.epsilon = 0.05
        self.discount = 0.8
        self.alpha = 0.2
        self.beliefsInitialized = False
        self.opponentMaybeDefinitePosition = {}

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.lastAction = None
        "Beliefs on opposing ghost positions."
        self.legalPositions = []
        walls = gameState.getWalls()
        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    self.legalPositions.append((x, y))

        CaptureAgent.registerInitialState(self, gameState)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
          # Only half a grid position was covered
          return successor.generateSuccessor(self.index, action)
        else:
          return successor

    def chooseAction(self, state):
        # Append game state to observation history...
        self.observationHistory.append(state)
        # Pick Action
        legalActions = state.getLegalActions(self.index)
        action = None
        if (DEBUG):
            print self.newline()
            print "AGENT " + str(self.index) + " choosing action!"
        if len(legalActions):
            if util.flipCoin(self.epsilon) and self.isTraining():
                action = random.choice(legalActions)
                if (DEBUG):
                    print "ACTION CHOSE FROM RANDOM: " + action
            else:
                action = self.computeActionFromQValues(state)
                if (DEBUG):
                    print "ACTION CHOSE FROM Q VALUES: " + action

        self.lastAction = action
        """ 
        TODO
        ReflexCaptureAgent has some code that returns to your side if there are less than 2 pellets
        We added that here
        """
        foodLeft = len(self.getFood(state).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for a in legalActions:
                successor = self.getSuccessor(state, a)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start,pos2)
                if dist < bestDist:
                    action = a
                    bestDist = dist

        if (DEBUG):
            print "AGENT " + str(self.index) + " chose action " + action + "!"
        return action

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        successor = self.getSuccessor(gameState, action)
        features = util.Counter()
        features['score'] = self.getScore(successor)
        if not self.red:
            features['score'] *= -1
        features['choices'] = len(successor.getLegalActions(self.index))
        return features

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        bestValue = -999999
        bestActions = None
        for action in state.getLegalActions(self.index):
            # For each action, if that action is the best then
            # update bestValue and update bestActions to be
            # a list containing only that action.
            # If the action is tied for best, then add it to
            # the list of actions with the best value.
            value = self.getQValue(state, action)
            if (DEBUG):
                print "ACTION: " + action + "           QVALUE: " + str(value)
            if value > bestValue:
                bestActions = [action]
                bestValue = value
            elif value == bestValue:
                bestActions.append(action)
        if bestActions == None:
            return Directions.STOP # If no legal actions return None
        return random.choice(bestActions) # Else choose one of the best actions randomly


    def getWeights(self):
        return self.weights

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        bestValue = -999999
        noLegalActions = True
        for action in state.getLegalActions(self.index):
            # For each action, if that action is the best then
            # update bestValue
            noLegalActions = False
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
        if noLegalActions:
            return 0 # If there is no legal action return 0
        # Otherwise return the best value found
        return bestValue

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        total = 0
        weights = self.getWeights()
        features = self.getFeatures(state, action)
        for feature in features:
            # Implements the Q calculation
            total += features[feature] * weights[feature]
        return total

    def getReward(self, gameState):
        foodList = self.getFood(gameState).asList()
        return -len(foodList)

    def observationFunction(self, gameState):
        if len(self.observationHistory) > 0 and self.isTraining():
            self.update(self.getCurrentObservation(), self.lastAction, gameState, self.getReward(gameState))
        if CHEAT:
            return gameState
        return gameState.makeObservation(self.index)

    def isTraining(self):
        return self.episodesSoFar < self.numTraining

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        if (DEBUG):
            print self.newline()
            print "AGENT " + str(self.index) + " updating weights!"
            print "Q VALUE FOR NEXT STATE: " + str(self.computeValueFromQValues(nextState))
            print "Q VALUE FOR CURRENT STATE: " + str(self.getQValue(state, action))
        difference = (reward + self.discount * self.computeValueFromQValues(nextState))
        difference -= self.getQValue(state, action)
        # Only calculate the difference once, not in the loop.
        newWeights = self.weights.copy()
        # Same with weights and features. 
        features = self.getFeatures(state, action)
        for feature in features:
            # Implements the weight updating calculations
            newWeight = newWeights[feature] + self.alpha * difference * features[feature]
            if (DEBUG):
                print "AGENT " + str(self.index) + " weights for " + feature + ": " + str(newWeights[feature]) + " ---> " + str(newWeight)
            newWeights[feature]  = newWeight
        self.weights = newWeights.copy()
        #print "WEIGHTS AFTER UPDATE"
        #print self.weights

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)
        if self.isTraining() and DEBUG:
            print "END WEIGHTS"
            print self.weights
        self.episodesSoFar += 1
        if self.episodesSoFar == self.numTraining:
            print "FINISHED TRAINING"

'''
Leeroy Agents
Time's up, let's do this
Agents that are meant to take the most efficient route possible
One agent that values pellets with a higher y coordinate more
One agent that values pellets with a lower y coordinate more
Eventually they will converge in the middle

TODOs that will fix this code, currently it loses every time to the baselineTeam
- make it go back to its own side every once in a while to 'cash in' the pellets it has
- account for that noisyDistance thing in our ghost distance formula
'''
def createTeam(firstIndex, secondIndex, isRed,
               first = 'LeeroyTopAgent', second = 'LeeroyBottomAgent', **args):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class LeeroyCaptureAgent(ApproximateQAgent):
  
  def registerInitialState(self, gameState):
    ApproximateQAgent.registerInitialState(self, gameState)
    self.favoredY = 0.0

  def __init__( self, index ):
  	ApproximateQAgent.__init__(self, index)
  	self.weights['successorScore'] = 100
  	self.weights['leeroyDistanceToFood'] = -1
  	self.weights['ghostDistance'] = 5
  	self.weights['stop'] = -1000
  	self.weights['legalActions'] = 100
  	self.weights['powerPelletValue'] = 100
  	self.distanceToTrackPowerPelletValue = 3
  	self.weights['backToSafeZone'] = -1
  	self.minPelletsToCashIn = 8
  	print "INITIAL WEIGHTS"
  	print self.weights

  def initializeBeliefs(self, gameState):
    self.beliefs = [None for x in range(len(self.getOpponents(gameState)) + len(self.getTeam(gameState)))]
    for opponent in self.getOpponents(gameState):
        self.initializeBelief(opponent)
    self.beliefsInitialized = True

  def initializeBelief(self, opponentIndex):
    belief = util.Counter()
    for p in self.legalPositions:
        belief[p] = 1.0
    belief.normalize()
    self.beliefs[opponentIndex] = belief

  def observeAllOpponents(self, gameState):
    if self.beliefsInitialized:
        for opponent in self.getOpponents(gameState):
            #print(gameState.getAgentPosition(opponent))
            self.observeOneOpponent(gameState, opponent)
    else: # Opponent indices are different in initialize() than anywhere else for some reason
        self.initializeBeliefs(gameState)

  def observeOneOpponent(self, gameState, opponentIndex):
    allPossible = util.Counter()
    allPossible[gameState.getAgentPosition(opponentIndex)] = 1
    self.beliefs[opponentIndex] = allPossible
    #self.displayDistributionsOverPositions(self.beliefs)
    self.opponentMaybeDefinitePosition[opponentIndex] = gameState.getAgentPosition(opponentIndex)
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food
    # uses leeroy distance so its prioritizes either top or bottom food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
        leeroyDistance = min([self.getLeeroyDistance(myPos, food) for food in foodList])
        features['leeroyDistanceToFood'] = leeroyDistance
      
    # If we are on our side
    onDefense = not myState.isPacman

    # Grab all non-scared enemy ghosts we can see
    self.observeAllOpponents(gameState)
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    nonScaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
    scaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
    if len(nonScaredGhosts) > 0:
        # Computes distance to enemy ghosts we can see
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in nonScaredGhosts]
        # Use the smallest distance
        smallestDist = min(dists)
        if smallestDist > 7:
            smallestDist = 0
        features['ghostDistance'] = smallestDist
    
    # If we are on defense and we are not scared, negate this value
    # So that we move closer to pacmen we can see
    # Doesn't work because ghosts are not pacmen
    #if onDefense and not myState.scaredTimer > 0:
        #features['ghostDistance'] = -features['ghostDistance']

    features['powerPelletValue'] = self.getPowerPelletValue(myPos, successor, scaredGhosts)
    
    # Heavily prioritize not stopping
    if action == Directions.STOP: 
        features['stop'] = 1
    
    # The total of the legalActions you can take from where you are AND
    # The legalActions you can take in all future states
    legalActions = gameState.getLegalActions(self.index)
    features['legalActions'] = len(legalActions)
    for legalAction in legalActions:
        newState = self.getSuccessor(gameState, legalAction).getAgentState(self.index)
        possibleNewActions = Actions.getPossibleActions( newState.configuration, gameState.data.layout.walls )
        features['legalActions'] += len(possibleNewActions)

    features['backToSafeZone'] = self.getCashInValue(myPos, gameState, myState)
    
    return features

  def getWeights(self):
    return self.weights

  def getLeeroyDistance(self, myPos, food):
      return self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1])

  def getPowerPelletValue(self, myPos, successor, scaredGhosts):
  	powerPellets = self.getCapsules(successor)
  	minDistance = 0
  	if len(powerPellets) > 0 and len(scaredGhosts) == 0:
		distances = [self.getMazeDistance(myPos, pellet) for pellet in powerPellets]
		minDistance = min(distances)
	return max(self.distanceToTrackPowerPelletValue - minDistance, 0)

  def getCashInValue(self, myPos, gameState, myState):
  	# if we have enough pellets, attempt to cash in
  	if myState.numCarrying >= self.minPelletsToCashIn:
  		return self.getMazeDistance(self.start, myPos)
  	else:
		return 0

# Leeroy Top Agent - favors pellets with a higher y
class LeeroyTopAgent(LeeroyCaptureAgent):
    
  def registerInitialState(self, gameState):
    LeeroyCaptureAgent.registerInitialState(self, gameState)
    self.favoredY = gameState.data.layout.height
    
# Leeroy Bottom Agent - favors pellets with a lower y
class LeeroyBottomAgent(LeeroyCaptureAgent):
    
  def registerInitialState(self, gameState):
    LeeroyCaptureAgent.registerInitialState(self, gameState)
    self.favoredY = 0.0