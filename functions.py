import numpy as np
import h5py

def initialization(n: int, d: int, r: list):
    rvalues = np.random.uniform(low=0, high=1, size=(n, d))
    particles = r[1] * rvalues + (1 - rvalues) * r[2]
    return particles

def init_fitness(x: np.array, n: int, f: callable):
    fit = np.zeros(shape=(1, n))
    for i in range(n):
        fit[i] = f(x[i, :])
    return fit

def init(n, dim, range, vmax, f):
    x = initialization(n=n, dim=dim, r=range)
    xfit = init_fitness(x=x, n=n, f=f)
    v = initialization(n=n, dim=dim, r=[-vmax, vmax])

    i = np.zeros(shape=(n, 1))
    c = np.zeros(shape=(n, 1))

    p = {
        "x": x,
        "xfit": xfit,
        "v": v,
        "p": p,
        "pfit": xfit,
        "oldxfit": xfit,
        "i": i,
        "c": c
    }

    return p

def get_best_global(p: dict):
    fit = np.min(p)
    i = np.argmin(p)
    g = {
        "x": p["x"][i, :],
        "xfit": fit
    }

    return g

def get_gradient(x0: np.ndarray, f: callable):
    g = np.zeros(x0.shape)
    fx0 = f(x0)
    step = 1e-5

    for i in range(0, len(x0)):
        xli = x0
        xli[i] = x0[i] + step
        g[i] = (f(xli) - fx0) / step
    
    return g

def trunc_grad(g: np.ndarray, gradmax: float):
    iddown = g < -gradmax
    idup = g > gradmax

    g[iddown] = -gradmax
    g[idup] = gradmax

    return g

def trunc_space(X, I, C, r):
    iddown = X < r[1]
    idup = X > r[2]

    if (any(iddown) == 1) or (any(idup) == 1):
        I = 1
        C = 0

    X[iddown] = r[1]
    X[idup] = r[2]

    return X, I, C

def update_best(x: np.ndarray, xfit: float, p: np.ndarray, pfit: float, g: dict):
    if xfit < pfit:
        p = x
        pfit = xfit
        if xfit < g["xfit"]:
            g["x"] = x
            g["xfit"] = xfit

    return p, pfit, g

def update_importance(x, I, fit, oldfit, c, g, gx, cmax, n):
    for i in range(n):
        if I[i] == 0:
            if (abs(fit[i] - oldfit[i]) < 1e-2) or (np.linalg.norm(g[i, :]) < 1e-2):
                c[i] = c[i] + 1
                if c[i] == cmax:
                    I[i] = 1
                    c[i] = 0
            else:
                c[i] = 0
        else:
            if np.sqrt(np.sum((x[i, :] - gx)**2)) < 1e-5:
                I[i] = 0
                c[i] = 0

    return I, c

def get_diversity(x: np.ndarray, l, n):
    avg = x.mean()
    d = np.sqrt(np.sum(np.sum(x - avg ** 2, axis=1))) / (n * l)

    return d

def update_direction(direction, diversity, I, t, n):
    if (direction > 0) and (diversity < t[1]):
        direction = -1
        I = np.ones(shape=(n, 1))
    elif (direction < 0) and (diversity > t[2]):
        direction = 1
        I = np.zeros(shape=(n, 1))

    return direction, I

