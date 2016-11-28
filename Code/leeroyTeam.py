from captureAgents import CaptureAgent
from baselineTeam import ReflexCaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Actions

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
               first = 'LeeroyTopAgent', second = 'LeeroyBottomAgent'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class LeeroyCaptureAgent(ReflexCaptureAgent):
  
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.favoredY = 0.0
  
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
        myPos = successor.getAgentState(self.index).getPosition()
        leeroyDistance = min([self.getLeeroyDistance(myPos, food) for food in foodList])
        features['leeroyDistanceToFood'] = leeroyDistance
      
    # If we are on our side
    onDefense = not myState.isPacman

    # Grab all non-scared enemy ghosts we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
    if len(ghosts) > 0:
        # Computes distance to enemy ghosts we can see
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
        # Use the smallest distance
        smallestDist = min(dists)
        # Only track it if the ghostDistance is smaller than X
        if smallestDist < 6:
            features['ghostDistance'] = smallestDist
    
    # If we are on defense and we are not scared, negate this value
    # So that we move closer to pacmen we can see
    # Doesn't work because ghosts are not pacmen
    #if onDefense and not myState.scaredTimer > 0:
        #features['ghostDistance'] = -features['ghostDistance']
    
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
    
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'leeroyDistanceToFood': -1, 'ghostDistance': 100, 'stop': -1000, 'legalActions': 100 }

  def getLeeroyDistance(self, myPos, food):
      return self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1])

# Leeroy Top Agent - favors pellets with a higher y
class LeeroyTopAgent(LeeroyCaptureAgent):
    
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.favoredY = gameState.data.layout.height
    
# Leeroy Bottom Agent - favors pellets with a lower y
class LeeroyBottomAgent(LeeroyCaptureAgent):
    
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.favoredY = 0.0