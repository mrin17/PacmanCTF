from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, math
from game import Directions
import game
from util import nearestPoint
from game import Actions

SONAR_NOISE_RANGE = 13 # Must be odd
SONAR_NOISE_VALUES = [i - (SONAR_NOISE_RANGE - 1)/2 for i in range(SONAR_NOISE_RANGE)]

# Shamelessly copied from capture.py to avoid messy imports
def getDistanceProb(trueDistance, noisyDistance):
    "Returns the probability of a noisy distance given the true distance"
    if noisyDistance - trueDistance in SONAR_NOISE_VALUES:
      return 1.0/SONAR_NOISE_RANGE
    else:
      return 0

def getObservationDistribution(noisyDistance):
    """
    Returns the factor P( noisyDistance | TrueDistances ), the likelihood of the provided noisyDistance
    conditioned upon all the possible true distances that could have generated it.
    """
    global observationDistributions
    if noisyDistance == None:
        return util.Counter()
    if noisyDistance not in observationDistributions:
        distribution = util.Counter()
        for error in SONAR_NOISE_VALUES:
            prob = getDistanceProb()
            distribution[max(1, noisyDistance - error)] += getDistanceProb(max(1, noisyDistance - error), noisyDistance)
        observationDistributions[noisyDistance] = distribution
    return observationDistributions[noisyDistance]

#################
# Team creation #
#################

DEBUG = False

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ApproximateQAgent', second = 'ApproximateQAgent', **args):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

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
        if len(self.observationHistory) > 0:
            self.observeAllOpponents(gameState)
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

    def newline(self):
        return "-------------------------------------------------------------------------"

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
                self.observeOneOpponent(gameState.getAgentPosition(opponent), gameState, opponent)
        else: # Opponent indices are different in initialize() than anywhere else for some reason
            self.initializeBeliefs(gameState)

    def getPositionDistribution(self, gameState, oldPos):
        pass

    def observeOneOpponent(self, observation, gameState, opponentIndex):
        allPossible = util.Counter()
        allPossible[gameState.getAgentPosition(opponentIndex)] = 1
        self.beliefs[opponentIndex] = allPossible
        #self.displayDistributionsOverPositions(self.beliefs)
        self.opponentMaybeDefinitePosition[opponentIndex] = gameState.getAgentPosition(opponentIndex)
        '''
        allPossible = util.Counter()
        for oldPos in self.legalPositions:
            # Probability a ghost is in a given position at time T = p(t)
            # Probability a ghost is in a given position at time T + 1 = p(t+1)
            # We want p(t+1), which is the sum of p(t+1 | t)*p(t) over p(T)
            oldProb = self.beliefs[opponentIndex][oldPos]
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            for newPos, prob in newPosDist.items():
                # prob is p(t+1 | t)
                # oldProb is p(t)
                # So multiply them to get p(t+1 | t) * p(t)
                change = prob * oldProb
                allPossible[newPos] += change
        allPossible.normalize()
        self.beliefs[opponentIndex] = allPossible
        '''

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