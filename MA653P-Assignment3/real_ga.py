import numpy as np

def sbx(p1, p2, bounds, eta=15):
    c1, c2 = p1.copy(), p2.copy()

    for i in range(len(p1)):
        if np.random.rand() <= 0.5:
            if abs(p1[i] - p2[i]) > 1e-14:
                x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                lb, ub = bounds[i]

                beta = 1 + (2*(x1 - lb)/(x2 - x1))
                alpha = 2 - beta**(-(eta+1))
                rand = np.random.rand()

                if rand <= 1/alpha:
                    betaq = (rand*alpha)**(1/(eta+1))
                else:
                    betaq = (1/(2 - rand*alpha))**(1/(eta+1))

                c1[i] = 0.5*((x1+x2) - betaq*(x2-x1))
                c2[i] = 0.5*((x1+x2) + betaq*(x2-x1))

                c1[i] = np.clip(c1[i], lb, ub)
                c2[i] = np.clip(c2[i], lb, ub)

    return c1, c2

def polynomial_mutation(x, bounds, pm=0.02, eta=20):
    for i in range(len(x)):
        if np.random.rand() < pm:
            lb, ub = bounds[i]
            delta1 = (x[i] - lb)/(ub - lb)
            delta2 = (ub - x[i])/(ub - lb)
            rand = np.random.rand()

            mut_pow = 1/(eta+1)

            if rand < 0.5:
                xy = 1 - delta1
                val = 2*rand + (1-2*rand)*(xy**(eta+1))
                deltaq = val**mut_pow - 1
            else:
                xy = 1 - delta2
                val = 2*(1-rand) + 2*(rand-0.5)*(xy**(eta+1))
                deltaq = 1 - val**mut_pow

            x[i] += deltaq*(ub - lb)
            x[i] = np.clip(x[i], lb, ub)

    return x

def tournament(pop, fitness):
    i, j = np.random.randint(len(pop), size=2)
    return pop[i] if fitness[i] < fitness[j] else pop[j]

def real_ga(f, bounds, pop_size=50, gens=200, pc=0.8, pm=0.02):
    pop = np.array([[np.random.uniform(b[0], b[1]) for b in bounds] for _ in range(pop_size)])

    for _ in range(gens):
        fitness = np.array([f(x) for x in pop])
        new_pop = []

        while len(new_pop) < pop_size:
            p1 = tournament(pop, fitness)
            p2 = tournament(pop, fitness)

            if np.random.rand() < pc:
                c1, c2 = sbx(p1, p2, bounds)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = polynomial_mutation(c1, bounds, pm)
            c2 = polynomial_mutation(c2, bounds, pm)

            new_pop.extend([c1, c2])

        pop = np.array(new_pop[:pop_size])

    return np.min([f(x) for x in pop])