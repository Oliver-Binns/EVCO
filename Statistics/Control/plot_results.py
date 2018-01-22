import pickle
import numpy
import matplotlib.pyplot as plt

means = []
maxs = []

for i in range(30):
    data = open("Run "+str(i)+".txt", "rb")
    log = pickle.load(data)

    mean = []
    max_fit = []
    for gen in log.chapters['fitness']:
        mean.append(gen['avg'])
        max_fit.append(gen['min'])

    means.append(mean)
    maxs.append(max_fit)
    

y1np = numpy.array(means)
y1 = y1np.mean(axis=0)
y2np = numpy.array(maxs)
y2 = y2np.mean(axis=0)

plt.plot(range(501), y1, label="Mean Fitness", color="b")
plt.plot(range(501), y2, label="Most Fit", color="g")
plt.ylabel("Fitness")
plt.xlabel("Generation")

plt.title("Fitness Progress through Generations")

plt.legend(loc='lower left')

#plt.savefig("fitness_benchmark.png")
plt.show()
