import numpy as np

BITS = 16

def decode(chrom, bounds):
    dim = len(bounds)
    decoded = []

    for i in range(dim):
        segment = chrom[i*BITS:(i+1)*BITS]
        val = int("".join(map(str, segment)), 2)
        low, high = bounds[i]
        real = low + (high-low)*val/(2**BITS - 1)
        decoded.append(real)

    return np.array(decoded)

def tournament(pop, fit):
    i, j = np.random.randint(len(pop), size=2)
    return pop[i] if fit[i] < fit[j] else pop[j]

def crossover(p1, p2, pc):
    if np.random.rand() < pc:
        point = np.random.randint(len(p1))
        child = np.concatenate([p1[:point], p2[point:]])
        return child
    return p1.copy()

def mutate(chrom, pm):
    for i in range(len(chrom)):
        if np.random.rand() < pm:
            chrom[i] ^= 1
    return chrom

def binary_ga(f, bounds, pop_size=50, gens=150, pc=0.8, pm=0.02):
    dim = len(bounds)
    pop = np.random.randint(2, size=(pop_size, dim*BITS))

    for _ in range(gens):
        fit = np.array([f(decode(c, bounds)) for c in pop])
        new_pop = []

        for _ in range(pop_size):
            p1 = tournament(pop, fit)
            p2 = tournament(pop, fit)
            child = crossover(p1, p2, pc)
            child = mutate(child, pm)
            new_pop.append(child)

        pop = np.array(new_pop)

    fit = np.array([f(decode(c, bounds)) for c in pop])
    return np.min(fit)