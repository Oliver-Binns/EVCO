# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
#import curses
import itertools
import numpy
import operator
import random
import time

from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools
from functools import partial

EVAL_RUNS = 10
POP_SIZE = 200
TOTAL_GENS = 700
MUT_PB = 0.2
CRX_PB = 0.5


S_RIGHT, S_LEFT, S_UP, S_DOWN = 0,1,2,3
XSIZE,YSIZE = 14,14
NFOOD = 1 # NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)

def if_then_else(condition, out1, out2):
	out1() if condition() else out2()
	
# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
	global S_RIGHT, S_LEFT, S_UP, S_DOWN
	global XSIZE, YSIZE

	def __init__(self):
		self.direction = S_RIGHT
		self.body = [
			[4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0]
		]
		self.score = 0
		self.ahead = []
		self.food = []

	def _reset(self):
		self.direction = S_RIGHT
		self.body[:] = [
			[4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0]
		]
		self.score = 0
		self.ahead = []
		self.food = []

	def getAheadLocation(self):
		self.ahead = [ self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1), self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)] 

	def updatePosition(self):
		self.getAheadLocation()
		self.body.insert(0, self.ahead )

	## You are free to define more sensing options to the snake
	def changeDirectionUp(self):
		self.direction = S_UP

	def changeDirectionRight(self):
		self.direction = S_RIGHT

	def changeDirectionDown(self):
		self.direction = S_DOWN

	def changeDirectionLeft(self):
		self.direction = S_LEFT

	def snakeHasCollided(self):
		self.hit = False
		if self.body[0][0] == 0 or self.body[0][0] == (YSIZE-1) or self.body[0][1] == 0 or self.body[0][1] == (XSIZE-1):
			self.hit = True
		if self.body[0] in self.body[1:]:
			self.hit = True
		return self.hit

	def sense_wall_ahead(self):
		self.getAheadLocation()
		return (
			self.ahead[0] == 0 or self.ahead[0] == (YSIZE-1) or self.ahead[1] == 0 
			or self.ahead[1] == (XSIZE-1)
		)

	def sense_food_ahead(self):
		self.getAheadLocation()
		return self.ahead in self.food

	def sense_tail_ahead(self):
		self.getAheadLocation()
		return self.ahead in self.body
		
	def form(self, method, out1, out2):
		return partial(if_then_else, method, out1, out2)
	
	def if_wall_ahead(self, out1, out2):
		return partial(if_then_else, self.sense_wall_ahead, out1, out2)
			
	def if_food_ahead(self, out1, out2):
		return partial(if_then_else, self.sense_food_ahead, out1, out2)
		
	def if_tail_ahead(self, out1, out2):
		return partial(if_then_else, self.sense_tail_ahead, out1, out2)
		
	def if_moving_right(self, out1, out2):
		return partial(if_then_else, lambda: self.direction == S_RIGHT, out1, out2)
		
	def if_moving_left(self, out1, out2):
		return partial(if_then_else, lambda: self.direction == S_LEFT, out1, out2)
		
	def if_moving_up(self, out1, out2):
		return partial(if_then_else, lambda: self.direction == S_UP, out1, out2)
	
	def if_moving_down(self, out1, out2):
		return partial(if_then_else, lambda: self.direction == S_DOWN, out1, out2)

# This function places a food item in the environment
def placeFood(snake):
	food = []
	while len(food) < NFOOD:
		potentialfood = [random.randint(1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
		if not (potentialfood in snake.body) and not (potentialfood in food):
			food.append(potentialfood)
	snake.food = food  # let the snake know where the food is
	return food


snake = SnakePlayer()


# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
def displayStrategyRun(individual):
	global snake
	global pset

	routine = gp.compile(individual, pset)

	curses.initscr()
	win = curses.newwin(YSIZE, XSIZE, 0, 0)
	win.keypad(1)
	curses.noecho()
	curses.curs_set(0)
	win.border(0)
	win.nodelay(1)
	win.timeout(120)

	snake._reset()
	food = placeFood(snake)

	for f in food:
		win.addch(f[0], f[1], '@')

	timer = 0
	collided = False
	while not collided and not timer == ((2*XSIZE) * YSIZE):
                time.sleep(0.3)
		# Set up the display
		win.border(0)
		win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')
 		win.getch()

		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##

		snake.updatePosition()

		if snake.body[0] in food:
			snake.score += 1
			for f in food: win.addch(f[0], f[1], ' ')
			food = placeFood(snake)
			for f in food: win.addch(f[0], f[1], '@')
			timer = 0
		else:    
			last = snake.body.pop()
			win.addch(last[0], last[1], ' ')
			timer += 1 # timesteps since last eaten
		win.addch(snake.body[0][0], snake.body[0][1], 'o')

		collided = snake.snakeHasCollided()
		hitBounds = (timer == ((2*XSIZE) * YSIZE))

	curses.endwin()
        print("Score:", snake.score)
	print("Collided: ", collided)
	print("Hit Bounds:", hitBounds)
	raw_input("Press to continue...")

	return snake.score,


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def runGame(instructions):
	global snake

	totalScore = 0

	snake._reset()
	food = placeFood(snake)
	timer = 0
	while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:

		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
		instructions()
		snake.updatePosition()

		if snake.body[0] in food:
			snake.score += 1
			food = placeFood(snake)
			timer = 0
		else:    
			snake.body.pop()
			timer += 1 # timesteps since last eaten

		totalScore += snake.score

	return totalScore,

#BASIC EA SET UP
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

pset = gp.PrimitiveSet("MAIN", 0)

def progn(*args):
	for arg in args:
		arg()

def prog2(out1, out2):
	return partial(progn, out1, out2)

# CURRENTLY WORKS ON boolean, can we use int values to improve this??
pset.addPrimitive(snake.if_tail_ahead, 2)
pset.addPrimitive(snake.if_wall_ahead, 2)
pset.addPrimitive(snake.if_food_ahead, 2)
pset.addPrimitive(snake.if_moving_up, 2)
pset.addPrimitive(snake.if_moving_down, 2)
pset.addPrimitive(snake.if_moving_right, 2)
pset.addPrimitive(snake.if_moving_left, 2)

pset.addTerminal(snake.changeDirectionUp)
pset.addTerminal(snake.changeDirectionDown)
pset.addTerminal(snake.changeDirectionLeft)
pset.addTerminal(snake.changeDirectionRight)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=4, max_=7)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def eval(code):
	tree = gp.compile(code, pset)
	total = 0

	for i in range(EVAL_RUNS):
		total += runGame(tree)[0]
	return total / EVAL_RUNS,


toolbox.register("evaluate", eval)

#toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selDoubleTournament,
	fitness_size=3, parsimony_size=1.2, fitness_first=False)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

for type in ["mate", "mutate"]:
	toolbox.decorate(
		type,
		gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
	)

def main():
	global snake
	global pset

	## THIS IS WHERE YOUR CORE EVOLUTIONARY ALGORITHM WILL GO #
	random.seed(318)

	pop = toolbox.population(n=POP_SIZE)
	hof = tools.HallOfFame(1)

	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(len)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", lambda val: round(numpy.mean(val), 2))
	mstats.register("std", lambda val: round(numpy.std(val), 2))
	mstats.register("min", numpy.min)
	mstats.register("max", numpy.max)

	pop, log = algorithms.eaSimple(
		pop,
		toolbox,
		CRX_PB,  # CHANCE OF CROSSOVER
		MUT_PB,  # CHANCE OF MUTATION
		TOTAL_GENS,  # NO Generations
		halloffame=hof,
		verbose=True,
		stats=mstats
	)

        
	# Total score as..
	# Attempt parsimony length prevention first
	best = tools.selBest(pop, 1)
	for ind in best:
                displayStrategyRun(ind)

        worst = tools.selWorst(pop, 1)
	for ind in worst:
                displayStrategyRun(ind)

main()
