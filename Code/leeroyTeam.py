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

DEBUG = False
DEFENSE_TIMER_MAX = 100.0

MINIMUM_PROBABILITY = .0001
beliefs = []
beliefsInitialized = []

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'LeeroyTopAgent', second = 'LeeroyBottomAgent', **args):
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

class LeeroyCaptureAgent(ApproximateQAgent):
	
	def registerInitialState(self, gameState):
		ApproximateQAgent.registerInitialState(self, gameState)
		self.favoredY = 0.0
		self.defenseTimer = 0.0
		self.lastNumReturnedPellets = 0.0
		self.getLegalPositions(gameState)

	def __init__( self, index ):
		ApproximateQAgent.__init__(self, index)
		self.weights = util.Counter()
		self.weights['successorScore'] = 100
		self.weights['leeroyDistanceToFood'] = -1
		self.weights['ghostDistance'] = 5
		self.weights['stop'] = -1000
		self.weights['legalActions'] = 100
		self.weights['powerPelletValue'] = 100
		self.distanceToTrackPowerPelletValue = 3
		self.weights['backToSafeZone'] = -1
		self.minPelletsToCashIn = 8
		self.weights['chaseEnemyValue'] = -100
		self.chaseEnemyDistance = 5
		# dictionary of (position) -> [action, ...]
		# populated as we go along; to use this, call self.getLegalActions(gameState)
		self.legalActionMap = {}
		self.legalPositionsInitialized = False
		if DEBUG:
			print "INITIAL WEIGHTS"
			print self.weights

	def getLegalPositions(self, gameState):
		if not self.legalPositionsInitialized:
			self.legalPositions = []
			walls = gameState.getWalls()
			for x in range(walls.width):
				for y in range(walls.height):
					if not walls[x][y]:
						self.legalPositions.append((x, y))
			self.legalPositionsInitialized = True
		return self.legalPositions

	def getLegalActions(self, gameState):
		"""
		legal action getter that favors 
		returns list of legal actions for Pacman in the given state
		"""
		currentPos = gameState.getAgentState(self.index).getPosition()
		if currentPos not in self.legalActionMap:
			self.legalActionMap[currentPos] = gameState.getLegalActions(self.index)
		return self.legalActionMap[currentPos]

	def getFeatures(self, gameState, action):
		self.observeAllOpponents(gameState)
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
		enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]
		nonScaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
		scaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
		if len(nonScaredGhosts) > 0:
			# Computes distance to enemy ghosts we can see
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in nonScaredGhosts]
			# Use the smallest distance
			smallestDist = min(dists)
			features['ghostDistance'] = smallestDist

		features['powerPelletValue'] = self.getPowerPelletValue(myPos, successor, scaredGhosts)
		features['chaseEnemyValue'] = self.getChaseEnemyWeight(myPos, enemyPacmen)
		
		# If we returned any pellets, we shift over to defense mode for a time
		if myState.numReturned != self.lastNumReturnedPellets:
			self.defenseTimer = DEFENSE_TIMER_MAX
			self.lastNumReturnedPellets = myState.numReturned
		# If on defense, heavily value chasing after enemies
		if self.defenseTimer > 0:
			self.defenseTimer -= 1
			features['chaseEnemyValue'] *= 100

		# If our opponents ate all our food (except for 2), we rush them
		if len(self.getFoodYouAreDefending(successor).asList()) <= 2:
			features['chaseEnemyValue'] *= 100

		# Heavily prioritize not stopping
		if action == Directions.STOP: 
				features['stop'] = 1
		
		# The total of the legalActions you can take from where you are AND
		# The legalActions you can take in all future states
		legalActions = self.getLegalActions(gameState)
		features['legalActions'] = len(legalActions)
		for legalAction in legalActions:
			newState = self.getSuccessor(gameState, legalAction)
			possibleNewActions = self.getLegalActions(newState)
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

	def getChaseEnemyWeight(self, myPos, enemyPacmen):
		if len(enemyPacmen) > 0:
			# Computes distance to enemy pacmen we can see
			dists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemyPacmen]
			# Use the smallest distance
			if len(dists) > 0:
				smallestDist = min(dists)
				return smallestDist
		return 0

	def initializeBeliefs(self, gameState):
		beliefs.extend([None for x in range(len(self.getOpponents(gameState)) + len(self.getTeam(gameState)))])
		for opponent in self.getOpponents(gameState):
			self.initializeBelief(opponent, gameState)
		beliefsInitialized.append('done')

	def initializeBelief(self, opponentIndex, gameState):
		belief = util.Counter()
		for p in self.getLegalPositions(gameState):
			belief[p] = 1.0
		belief.normalize()
		beliefs[opponentIndex] = belief

	def observeAllOpponents(self, gameState):
		if len(beliefsInitialized):
			for opponent in self.getOpponents(gameState):
				self.observeOneOpponent(gameState, opponent)
		else: # Opponent indices are different in initialize() than anywhere else for some reason
			self.initializeBeliefs(gameState)
		self.displayDistributionsOverPositions(beliefs)

	def observeOneOpponent(self, gameState, opponentIndex):
		noisyDistance = gameState.getAgentDistances()[opponentIndex]
		pacmanPosition = gameState.getAgentPosition(self.index)
		allPossible = util.Counter()
		# We might have a definite position for the agent - if so, no need to do calcs
		maybeDefinitePosition = gameState.getAgentPosition(opponentIndex)
		if maybeDefinitePosition != None:
			allPossible[maybeDefinitePosition] = 1
			beliefs[opponentIndex] = allPossible
			return
		for p in self.getLegalPositions(gameState):
			# For each legal ghost position, calculate distance to that ghost
			trueDistance = util.manhattanDistance(p, pacmanPosition)
			modelProb = gameState.getDistanceProb(trueDistance, noisyDistance) # Find the probability of getting this noisyDistance if the ghost is at this position
			if modelProb > 0:
				# We'd like to find the probability of the ghost being at this distance
				# given that we got this noisy distance
				# p(noisy | true) = p(true | noisy) * p(true) / p(noisy)
				# p(noisy) is 1 - we know that for certain.
				# So return p(true | noisy) * p(true)
				oldProb = beliefs[opponentIndex][p]
				# Add a small constant to oldProb because a ghost may travel more than
				# 13 spaces - if that happens then we don't want to think it's prob is 0
				allPossible[p] = (oldProb + MINIMUM_PROBABILITY) * modelProb
			else:
				allPossible[p] = 0
		allPossible.normalize()
		beliefs[opponentIndex] = allPossible

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