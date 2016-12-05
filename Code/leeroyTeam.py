from captureAgents import CaptureAgent
from baselineTeam import ReflexCaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Actions
from qLearningAgent import ApproximateQAgent

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
  	self.weights['backToStartDistance'] = -1000
  	self.threatenedDistance = 3
  	self.weights['powerPelletValue'] = 100
  	self.distanceToTrackPowerPelletValue = 3
  	self.weights['backToSafeZone'] = -1
  	self.minPelletsToCashIn = 8
  	print "INITIAL WEIGHTS"
  	print self.weights
  
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
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    nonScaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
    scaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
    if len(nonScaredGhosts) > 0:
        # Computes distance to enemy ghosts we can see
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in nonScaredGhosts]
        # Use the smallest distance
        smallestDist = min(dists)
        features['ghostDistance'] = smallestDist
        features['backToStartDistance'] = self.getBackToStartDistance(myPos, smallestDist)
    
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

  def getBackToStartDistance(self, myPos, smallestGhostDist):
  	if smallestGhostDist > self.threatenedDistance:
  		return 0
  	else:
  		return self.getMazeDistance(self.start,myPos)

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