from captureAgents import CaptureAgent
from baselineTeam import ReflexCaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Actions

'''
Ambush Agents
Will seek out and try to ambush any enemies.
Currently does not care for pellets, because it has a pretty high chance
of dying anyways. 
If it's pretty close to a non-scared ghost on the enemy side (mod. your
	definition of "pretty close"), it'll go for a power pellet, though.

Based on LeeroyAgent, so this probably still applies:
TODOs that will fix this code, currently it loses every time to the baselineTeam
- make it go back to its own side every once in a while to 'cash in' the pellets it has
- account for that noisyDistance thing in our ghost distance formula
'''
def createTeam(firstIndex, secondIndex, isRed,
               first = 'AmbushTopAgent', second = 'AmbushBottomAgent'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class AmbushCaptureAgent(ReflexCaptureAgent):
  
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
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
        myPos = successor.getAgentState(self.index).getPosition()
        dist = min([self.getAmbushDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = dist
      
    # If we are on our side
    onDefense = not myState.isPacman

    # Grab all enemy ghosts we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    pacmen = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
    scaredGhosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None and enemy.scaredTimer > 0]

    # need a default in case there are no applicable things for the min
    defaultDistance = -1000
    # in order of priority:
    # track distance to nearest pacman (implicitly on our side)
    features['pacmanDistance'] = min([self.getMazeDistance(myPos, enemy.getPosition()) for enemy in pacmen] or [defaultDistance])
    # track distance to nearest scared ghost
    features['scaredDistance'] = min([self.getMazeDistance(myPos, enemy.getPosition()) for enemy in scaredGhosts] or [defaultDistance])
    # track distance to the nearest non-scared ghost
    features['ghostDistance'] = min([self.getMazeDistance(myPos, enemy.getPosition()) for enemy in ghosts] or [defaultDistance])

    # track distance to nearest power pellet
    # TODO: do we need to filter out the ones on our side?
    pellets = self.getCapsules(myState)
    if not scaredGhosts:
        features['capsuleValue'] = min([self.getMazeDistance(myPos, capsulePos) for capsulePos in pellets] or [0])
    else:
        features['capsuleValue'] = 0 # don't want to eat two really quickly
    
    # if we're on defense and scared, maybe we don't actually want to prioritize that pacman
    # TODO: should that be a not??
    if onDefense and myState.scaredTimer > 0:
        features['pacmanDistance'] = -features['pacmanDistance']

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
    return {'successorScore': 50, 'distanceToFood': -1, 'ghostDistance': 75, 'pacmanDistance': 100, 'scaredDistance': 90, 'capsuleValue': 100,
            'stop': -1000, 'legalActions': 10 }

  def getAmbushDistance(self, myPos, food):
      return self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1])

# Top Agent - favors pacmen with a higher y
class AmbushTopAgent(AmbushCaptureAgent):
    
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.favoredY = gameState.data.layout.height
    
# Bottom Agent - favors pacmen with a lower y
class AmbushBottomAgent(AmbushCaptureAgent):
    
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.favoredY = 0.0