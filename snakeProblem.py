# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
#import console
import itertools
import multiprocessing
import numpy
import operator
import random
import time
from timeit import default_timer as timer

import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools
from functools import partial

EVAL_RUNS = 3
POP_SIZE = 5000
TOTAL_GENS = 250
MUT_PB = 0.1
CRX_PB = 0.8


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
		self.ahead = self.getAheadOf(self.body[0])

	def getAheadOf(self, cell):
		return [
			cell[0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1),
			cell[1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)
		]

	def updatePosition(self):
		self.getAheadLocation()
		self.body.insert(0, self.ahead)

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
		if self.body[0][0] == 0 or self.body[0][0] == (YSIZE-1) or self.body[0][1] == 0 or self.body[0][1] == (XSIZE-1): return True
		if self.body[0] in self.body[1:]: return True
		return False

	def is_wall(self, cell):
		return (
			cell[0] == 0 or cell[0] == (YSIZE-1) or
			cell[1] == 0 or cell[1] == (XSIZE-1)
		)

	#if this cell is part of the snake excluding IT'S HEAD
	def is_tail(self, cell):
		return cell in self.body[1:]

	#wall must be IMMEDIATELY ahead to return true
	def sense_wall_ahead(self):
		self.getAheadLocation()
		return self.is_wall(self.ahead)

	#tail must be IMMEDIATELY ahead to return true!
	def sense_tail_ahead(self):
		self.getAheadLocation()
		return is_tail(self.ahead)

	def sense_danger_left(self):
		self.getAheadLocation()
		left = list(self.body[0])
		if self.direction == S_UP:
			left[0] -= 1
		elif self.direction == S_DOWN:
			left[0] += 1
		elif self.direction == S_RIGHT:
			left[1] += 1
		elif self.direction == S_LEFT:
			left[1] -= 1
		return self.is_tail(left) or self.is_wall(left)

	def sense_danger_right(self):
		self.getAheadLocation()
		right = list(self.body[0])

		if self.direction == S_UP:
			right[0] += 1
		elif self.direction == S_DOWN:
			right[0] -= 1
		elif self.direction == S_RIGHT:
			right[1] -= 1
		elif self.direction == S_LEFT:
			right[1] += 1
		return self.is_tail(right) or self.is_wall(right)

	#tail is ahead in any cell along this row!
	def sense_tail_before_food(self):
		self.getAheadLocation()

		#up/down is FIRST coordinate
		#origin (0, 0) is top/left
		for item in self.food:
			if item[0] == self.ahead[0]:
				if self.direction == S_LEFT or self.direction == S_RIGHT:
					for x in range(self.ahead[1], item[1]):
						if [item[0], x] in self.body[1:]:
							return True

			if item[1] == self.ahead[1]:
				if self.direction == S_UP or self.direction == S_DOWN:
					for y in range(self.ahead[0], item[0]):
						if [y, item[1]] in self.body[1:]:
							return True

		return False

	#food can be ahead in any cell along this row!
	def sense_food_ahead(self):
		self.getAheadLocation()

		#up/down is FIRST coordinate
		#origin (0, 0) is top/left
		for item in self.food:
			if self.direction == S_UP:
				if (item[0] <= self.ahead[0] and
				   item[1] == self.ahead[1]):
					return True

			elif self.direction == S_DOWN:
				if (item[0] >= self.ahead[0] and
					item[1] == self.ahead[1]):
					return True

			elif self.direction == S_RIGHT:
				if (item[0] == self.ahead[0] and
					item[1] >= self.ahead[1]):
					return True

			elif self.direction == S_LEFT:
				if (item[0] == self.ahead[0] and
					item[1] <= self.ahead[1]):
					return True
		return False

	def sense_danger_two_ahead(self):
		self.getAheadLocation()
		two_ahead = self.getAheadOf(self.ahead)

		return self.is_tail(two_ahead) or self.is_wall(two_ahead)

	#proximity sensing
	def if_danger_ahead(self, out1, out2):
		return partial(if_then_else, lambda: self.sense_wall_ahead or sense_tail_ahead, out1, out2)

	def if_danger_left(self, out1, out2):
		return partial(if_then_else, self.sense_danger_left, out1, out2)

	def if_danger_right(self, out1, out2):
		return partial(if_then_else, self.sense_danger_right, out1, out2)

	def if_tail_before_food(self, out1, out2):
		return partial(if_then_else, self.sense_tail_before_food, out1, out2)

	def if_food_ahead(self, out1, out2):
		return partial(if_then_else, self.sense_food_ahead, out1, out2)

	def if_danger_two_ahead(self, out1, out2):
		return partial(if_then_else, lambda: self.sense_danger_two_ahead, out1, out2)

	#food directionality check
	def if_food_up(self, out1, out2):
		return partial(if_then_else, lambda: self.body[0][0] > self.food[0][0], out1, out2)

	def if_food_down(self, out1, out2):
		return partial(if_then_else, lambda: self.body[0][0] < self.food[0][0], out1, out2)

	def if_food_left(self, out1, out2):
		return partial(if_then_else, lambda: self.body[0][1] > self.food[0][1], out1, out2)

	def if_food_right(self, out1, out2):
		return partial(if_then_else, lambda: self.body[0][1] < self.food[0][1], out1, out2)

	#movement checks- to compensate for different terminal functionality in this variant
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
		routine()
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

def displayRunPythonista(individual):
	global snake
	global pset

	routine = gp.compile(individual, pset)

	snake._reset()
	food = placeFood(snake)

	timer = 0
	collided = False

	endline = "-" * (XSIZE + 2)

	while not collided and not timer == ((2*XSIZE) * YSIZE):
		time.sleep(0.3)
		# Set up the display
		console.clear()
		print(0, 2, 'Score : ' + str(snake.score) + ' ')
		print endline
		for x in range(XSIZE):
			line = "|"
			for y in range(YSIZE):
				if [x, y] in snake.body:
					line += "X"
				elif [x, y] in food:
					line += "@"
				else:
					line += " "
			print line + "|"
		print endline

		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
		routine()
		snake.updatePosition()

		if snake.body[0] in food:
			snake.score += 1
			food = placeFood(snake)
			timer = 0
		else:
			last = snake.body.pop()
			timer += 1 # timesteps since last eaten

		collided = snake.snakeHasCollided()
		hitBounds = (timer == ((2*XSIZE) * YSIZE))

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
def runGame(individual):
	global snake
	global pset

	routine = gp.compile(individual, pset)

	snake._reset()
	food = placeFood(snake)
	timer = 0

	while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:
		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
		routine()
		snake.updatePosition()

		if snake.body[0] in food:
			snake.score += 1
			food = placeFood(snake)
			timer = 0
		else:
			snake.body.pop()
			timer += 1 # timesteps since last eaten

	return snake.score,

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

#allows multiple elements to be computed after each other..
pset.addPrimitive(prog2, 2)

pset.addPrimitive(snake.if_danger_ahead, 2)
pset.addPrimitive(snake.if_danger_left, 2)
pset.addPrimitive(snake.if_danger_right, 2)
pset.addPrimitive(snake.if_food_ahead, 2)
pset.addPrimitive(snake.if_tail_before_food, 2)
pset.addPrimitive(snake.if_danger_two_ahead, 2)


pset.addPrimitive(snake.if_food_up, 2)
pset.addPrimitive(snake.if_food_down, 2)
pset.addPrimitive(snake.if_food_left, 2)
pset.addPrimitive(snake.if_food_right, 2)

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

BOARD_SIZE = XSIZE * YSIZE
MAX_FITNESS = BOARD_SIZE - len(snake.body)
def eval(individual):
	total = 0
	for i in range(EVAL_RUNS):
		total += runGame(individual)[0]
	return MAX_FITNESS - (total / EVAL_RUNS),

def fib(n):
	if n == 0 or n == 1:
		return 1
	return fib(n) + fib(n - 1)

#MULTIPLE EVALUATION FUNCTIONS TO COMPARE
def evaluate_score(individual):
	total = 0
	for i in range(EVAL_RUNS):
		total += runGame(individual)[0]
	return (total / EVAL_RUNS),

def evaluate_score_square(individual):
	total = 0
	for i in range(EVAL_RUNS):
		total += runGame(individual)[0] ** 2
	return (total / EVAL_RUNS),

def evaluate_factorial(individual):
	total = 0
	for i in range(EVAL_RUNS):
		total += runGame(individual)[0] * (total if total > 0 else 1)
	return (total / EVAL_RUNS),


toolbox.register("evaluate", evaluate_factorial)

toolbox.register("select", tools.selTournament, tournsize=3)
#toolbox.register("select", tools.selDoubleTournament,
#	fitness_size=3, parsimony_size=1.2, fitness_first=False)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

for type in ["mate", "mutate"]:
	toolbox.decorate(
		type,
		gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
	)

def main():
	global snake
	global pset

	## THIS IS WHERE YOUR CORE EVOLUTIONARY ALGORITHM WILL GO #
	#random.seed(318)
	pop = toolbox.population(n=POP_SIZE)
	hof = tools.HallOfFame(5)

	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(len)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", lambda val: round(numpy.mean(val), 2))
	mstats.register("std", lambda val: round(numpy.std(val), 2))
	mstats.register("min", numpy.min)
	mstats.register("max", numpy.max)

	start = timer()

	try:
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
	except KeyboardInterrupt:
		pool.terminate()
		pool.join()
		raise KeyboardInterrupt

	end = timer()

	# Total score as..
	# Attempt parsimony length prevention first
	best = tools.selBest(pop, 1)
	for ind in best:
		#runs = []
		for run in range(500):
			displayStrategyRun(ind)
			#runs.append(runGame(ind)[0])

		#time_log = open("Statistics/Population/Fitness.txt", "a+")
		#time_log.write("Population Size: " + str(POP_SIZE) + "\n")
		#time_log.write("Run " + str(i) + "\n")
		#time_log.write("Elapsed: " + str(end - start))
		#time_log.write(str(runs) + "\n")
		#time_log.write(str(max(runs)) + "\n")
		#time_log.write(str(numpy.mean(runs)) + "\n")
		#time_log.write(str(numpy.std(runs)) + "\n\n")
		#time_log.close()
		#print(runs)
		#print(max(runs))
		#print("Elapsed: " + str(end - start))
		#print(numpy.mean(runs))
		#print(numpy.std(runs))
		#print
		#print(runGame(ind))



main()
