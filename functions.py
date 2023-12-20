import numpy as np

from cec17 import cec2017

def initialization(n: int, d: int, r: list):
    rvalues = np.random.uniform(low=0, high=1, size=(n, d))
    particles = r[0] * rvalues + (1 - rvalues) * r[1]
    return particles

def init_fitness(x: np.array, n: int, idf: int):
    fit = np.zeros(shape=(1, n))

    for i in range(n):
        fit[i] = cec2017(x=x[i, :], fun_nums=idf)

    return fit

def init(n, dim, range, vmax, idf):
    x = initialization(n=n, d=dim, r=range)
    xfit = init_fitness(x=x, n=n, idf=idf)
    v = initialization(n=n, d=dim, r=[-vmax, vmax])

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

def get_gradient(x0: np.ndarray, idf: int):
    g = np.zeros(x0.shape)
    fx0 = cec2017(x=x0, fun_nums=idf)
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

def m_function(p):
    a = -0.5 + np.random.uniform(p["d"])
    m = p["i"] + p["a"] * (a - np.transpose(a))

    return m

def get_velocity(x, v, p, s, gb, I, params, direction):
    if dir == 1:
        pt1 = I * params["cc"] * np.random.uniform(0, 1, 1) * (p - x)
        pt2 = I * params["sc"] * np.random.uniform(0, 1, 1) * (gb["x"] - x)
        pt3 = (I-1) * params["gc"] * np.random.uniform(0, 1, 1) * s
        
        v = params["iw"] * v + direction * (pt1 + pt2 + pt3)
    else:
        m1 = m_function(params)
        m2 = m_function(params)

        pt1 = I * params["sc"] * np.random.uniform(0, 1, 1) * np.transpose(m1 * np.transpose(gb["x"] - x))
        pt2 = (I-1) * params["gc"] * np.random.uniform(0, 1, 1) * np.transpose(m2 * np.transpose(s))

        v = params["iw"] * v + direction * (pt1 + pt2)

    return v

def trunc_vel(v, vmax):
    iddown = v < -vmax
    idup = v > vmax

    v[iddown] = -vmax
    v[idup] = vmax

    return v

def trunc_space(x, i, c, r):
    iddown = x < r[1]
    idup = x > r[2]

    if (any(iddown) == 1) or (any(idup) == 1):
        i = 1
        c = 0

    x[iddown] = r[1]
    x[idup] = r[2]

    return x, i, c

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

