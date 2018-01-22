import ast
import numpy

data = open("250/Timing.txt", "r");

scores = []
value = None

for line in data:
    if "Scores: " in line:
        value = line.replace("Scores: ", "")
    elif value is not None:
        if line.isspace():
            scores.append(ast.literal_eval(value))
            value = None
        else:
            value += line

maxs = []
means = []
stdevs = []

for run in scores:
    maxs.append(max(run))
    means.append(numpy.mean(run))
    stdevs.append(numpy.std(run))

print(numpy.mean(maxs))
print(numpy.mean(means))
print(numpy.mean(stdevs))
