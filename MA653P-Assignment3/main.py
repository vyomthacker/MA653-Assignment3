import numpy as np
from functions import *
from gradient_opt import multiple_restarts
from binary_ga import binary_ga
from real_ga import real_ga

def stats(arr):
    return np.min(arr), np.max(arr), np.mean(arr), np.std(arr)

runs = 30

# --- Gradient ---
print("\nGradient Results")
print("Rastrigin:", stats(multiple_restarts(rastrigin, [(-5.12,5.12)]*2, runs)))
print("Rosenbrock:", stats(multiple_restarts(rosenbrock, [(-3,3)]*2, runs)))
print("Himmelblau:", stats(multiple_restarts(himmelblau, [(-6,6)]*2, runs)))

# --- Binary GA ---
print("\nBinary GA")
print("Rastrigin:", stats([binary_ga(rastrigin, [(-5.12,5.12)]*2) for _ in range(runs)]))
print("Rosenbrock:", stats([binary_ga(rosenbrock, [(-3,3)]*2) for _ in range(runs)]))
print("Himmelblau:", stats([binary_ga(himmelblau, [(-6,6)]*2) for _ in range(runs)]))

# --- Real GA ---
print("\nReal GA")
print("Himmelblau:", stats([real_ga(himmelblau, [(-6,6)]*2) for _ in range(runs)]))

# --- Constrained GA ---
bounds = [(0,6), (0,6)]

print("\nConstrained Problem")
print("Binary GA:", stats([binary_ga(constrained_objective, bounds) for _ in range(runs)]))
print("Real GA:", stats([real_ga(constrained_objective, bounds) for _ in range(runs)]))