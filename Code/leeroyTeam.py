from captureAgents import CaptureAgent
from baselineTeam import ReflexCaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

'''
Leeroy Agents
Time's up, let's do this
Agents that are meant to take the most efficient route possible
One agent that values pellets with a higher y coordinate more
One agent that values pellets with a lower y coordinate more
Eventually they will converge in the middle

TODOs that will fix this code, currently it loses every time to the baselineTeam
- make it actually try to avoid ghosts
- make it go back to its own side every once in a while to 'cash in' the pellets it has
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
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      leeroyDistance = min([self.getLeeroyDistance(myPos, food) for food in foodList])
      features['leeroyDistanceToFood'] = leeroyDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'leeroyDistanceToFood': -1 }

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