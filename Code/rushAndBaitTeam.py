from captureAgents import CaptureAgent
from baselineTeam import ReflexCaptureAgent
from leeroyTeam import LeeroyCaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

'''
DO NOT USE THIS!!! IT IS BAD :(
Rush And Bait Team
AKA Russian Bait
AKA Project Arstotzka
Glory to Arstotzka
Unique and Niche strategy that is so bold it might just work
Step 1) Rush a Power Pellet
Step 2) Grab as many dots as you can while the ghosts are white
Step 3) Repeat Steps 1 and 2 until there are no more Power Pellets
Step 4) Agent 1 RUSHES the remaining pellets
        Agent 2 BAITS any opposing defender that is closest to Agent 1 and tries to keep them off Agent 1

Code we need to take into account
- What if the other agent is hyper aggressive?
'''
def createTeam(firstIndex, secondIndex, isRed,
               first = 'RushAgent', second = 'BaitAgent'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class RussianCaptureAgent(LeeroyCaptureAgent):
  
  def registerInitialState(self, gameState):
    LeeroyCaptureAgent.registerInitialState(self, gameState)
    self.rushesRemainingPellets = True
  
  def getFeatures(self, gameState, action):
    features = LeeroyCaptureAgent.getFeatures(self, gameState, action)
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    
    friendIndices = [a for a in self.getTeam(gameState) if not a == self.index]
    friendState = successor.getAgentState(friendIndices[0])
    friendPos = friendState.getPosition()
    
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    scaredGhosts = [a for a in ghosts if a.scaredTimer > 0]
    features['distanceToPowerPellet'] = 0
    features['distanceToFriend'] = self.getMazeDistance(myPos, friendPos)
    
    powerPellets = self.getCapsules(successor)
    if len(powerPellets) > 0:
        if len(scaredGhosts) == 0:
            distances = [self.getMazeDistance(myPos, pellet) for pellet in powerPellets]
            features['distanceToPowerPellet'] = min(distances)
    else:
        if not self.rushesRemainingPellets:
            # get teammate
            # see which ghost is closest to that teammate
            bestGhost = None
            for ghost in ghosts:
                if bestGhost == None or self.getMazeDistance(friendPos, ghost.getPosition()) < self.getMazeDistance(friendPos, bestGhost.getPosition()):
                    bestGhost = ghost
            if not bestGhost == None:
                distToThatGhost = self.getMazeDistance(friendPos, bestGhost.getPosition())
                # try to bait that ghost by moving close to it
                features['distanceToGhostAttackingPartner'] = distToThatGhost
            # otherwise, prioritize pellets
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1000, 'leeroyDistanceToFood': -10, 
            'ghostDistance': 1000, 'stop': -10000, 'legalActions': 1000, 
            'distanceToPowerPellet': -100, 'distanceToGhostAttackingPartner' : -2000, 'distanceToFriend': 1 }



class RushAgent(RussianCaptureAgent):
    
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.favoredY = gameState.data.layout.height
    self.rushesRemainingPellets = True

class BaitAgent(RussianCaptureAgent):
    
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.favoredY = 0.0
    self.rushesRemainingPellets = False