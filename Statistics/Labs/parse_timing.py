import ast
import numpy
import matplotlib.pyplot as plt

all_times = []
all_scores = []

POPS = [100, 250, 500, 750, 1000, 2500, 5000, 10000]

for POP_SIZE in POPS:

    data = open(str(POP_SIZE) + "/Timing.txt", "r");

    times = []
    scores = []
    value = None

    for line in data:
        if "Elapsed: " in line:
            times.append(ast.literal_eval(line.replace("Elapsed: ", "").replace("\n", "")))
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

    print("Times:")
    print("Mean: " + str(numpy.mean(times)))
    print("Max: " + str(numpy.max(times)))
    print("St.Dev: " +  str(numpy.std(times)))
    print()
    print("Results:")
    print("Mean: " + str(numpy.mean(means)))
    print("Max: " + str(numpy.mean(maxs)))
    print("St.Dev: " +  str(numpy.mean(stdevs)))

    all_times.append(times)
    all_scores.append(numpy.mean(means))

y1np = numpy.array(all_times)
y1 = y1np.mean(axis=1)

fig, axis = plt.subplots()
axis2 = axis.twinx()
axis.plot(POPS, y1, label="Run Times", color="g")
axis.set_ylabel("Time (Seconds)")
axis2.plot(POPS, all_scores, label="Mean Fitness", color="b")
axis2.set_ylabel("Fitness")

plt.title("Mean Score and Run Time for Various Population Sizes")

plt.xlabel("Population Size")
plt.xscale('log')
plt.legend(loc='lower right')

def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None
color_y_axis(axis, 'g')
color_y_axis(axis2, 'b')

#plt.savefig("fitness_benchmark.png")
plt.show()
