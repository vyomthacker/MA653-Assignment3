import numpy as np

def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x1[i] += eps
        grad[i] = (f(x1) - f(x)) / eps
    return grad

def gradient_descent(f, bounds, lr=0.001, iters=1000):
    x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])

    for _ in range(iters):
        grad = numerical_gradient(f, x)
        x = x - lr * grad
        # clip to bounds
        for i in range(len(bounds)):
            x[i] = np.clip(x[i], bounds[i][0], bounds[i][1])

    return f(x)

def multiple_restarts(f, bounds, runs=30):
    return [gradient_descent(f, bounds) for _ in range(runs)]