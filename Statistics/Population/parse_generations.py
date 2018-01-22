import pickle
import numpy
import matplotlib.pyplot as plt

all_lines = []

POPS = [100, 250, 500, 750, 1000, 2500, 5000, 10000]

for POP_SIZE in POPS:
    avg_fitness = []
    
    for i in range(30):
        data = open(str(POP_SIZE) + "/Run " + str(i) + ".txt", "rb")
        log = pickle.load(data, encoding="latin1")
        
        run_fit = []
        for gen in log.chapters['fitness']:
            run_fit.append(gen["avg"])
        
        avg_fitness.append(run_fit)        

    y1np = numpy.array(avg_fitness)
    y1 = y1np.mean(axis=0)

    all_lines.append(y1)   

#y1np = numpy.array(all_times)
#y1 = y1np.mean(axis=1)
print(len(all_lines))
for index, line in enumerate(all_lines):
    plt.plot(range(251), line, label=str(POPS[index]))
    
plt.ylabel("Average Fitness")

plt.title("Generations needed to Converge")

plt.xlabel("Generation")
#plt.xscale('log')
plt.legend(loc='lower left')

#plt.savefig("fitness_benchmark.png")
plt.show()
