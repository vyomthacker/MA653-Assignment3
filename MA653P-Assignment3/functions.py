import numpy as np

# --- Unconstrained functions ---

def rastrigin(x):
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def rosenbrock(x):
    return sum(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


# --- Constrained problem ---

def constrained_f(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def g1(x):
    x1, x2 = x
    return 4.84 - (x1 - 0.05)**2 - (x2 - 2.5)**2

def g2(x):
    x1, x2 = x
    return x1**2 + (x2 - 2.5)**2 - 4.84

def penalty(x):
    return max(0, -g1(x)) + max(0, -g2(x))

def constrained_objective(x):
    return constrained_f(x) + 1e6 * penalty(x)