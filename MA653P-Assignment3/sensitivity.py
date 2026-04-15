import numpy as np
import matplotlib.pyplot as plt
from binary_ga import binary_ga
from functions import rastrigin

mutation_probs = [0.01, 0.02, 0.05]
crossover_probs = [0.7, 0.8, 0.9]

runs = 30

results = np.zeros((len(mutation_probs), len(crossover_probs)))

for i, pm in enumerate(mutation_probs):
    for j, pc in enumerate(crossover_probs):
        errors = []

        for _ in range(runs):
            val = binary_ga(rastrigin, [(-5.12,5.12)]*2, pc=pc, pm=pm)
            errors.append(val)

        results[i, j] = np.mean(errors)

print("Average Error Table:")
print(results)


plt.figure()

plt.imshow(results)
plt.colorbar(label="Average Error")

plt.xticks(range(len(crossover_probs)), crossover_probs)
plt.yticks(range(len(mutation_probs)), mutation_probs)

plt.xlabel("Crossover Probability")
plt.ylabel("Mutation Probability")
plt.title("Parameter Sensitivity (Binary GA)")

plt.savefig("sensitivity_heatmap.png")
plt.show()


plt.figure()

for i, pm in enumerate(mutation_probs):
    plt.plot(crossover_probs, results[i], marker='o', label=f"pm={pm}")

plt.xlabel("Crossover Probability")
plt.ylabel("Average Error")
plt.title("Effect of Crossover for Different Mutation Rates")
plt.legend()

plt.savefig("sensitivity_lines.png")
plt.show()