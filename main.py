import numpy as np
import scipy.io

from cec17 import cec17
from functions import *

def main(idf, t, d, stopc, gc):

    g = scipy.io.loadmat("g_opt.mat")
    g_opt = g["g_opt"]
    rnge = [-100, 100]

    n = 20
    k = 1
    cmax = 3
    dt = [1e-2, 0.25]
    direction = 1
    alpha = 3

    params = {
        "IW": 0.7298,
        "CC": 0,
        "SC": 1.4962,
        "GC": gc,
        "d": d,
        "a": alpha * np.pi / 180,
        "i": np.eye(d)
    }

    l = np.linalg.norm(np.ones(shape=(1, d)) * (rnge[1] - rnge[0]))
    vmax = k * (rnge[1] - rnge[0]) / 2

    p = init(n, d, rnge, vmax, idf)
    gbest = get_best_global(p)

    sr = 0

    for i in range(t):
        for j in range(n):
            if p["i"][j] == 0:
                p["g"][j, :] = get_gradient(p["x"][j, :], idf)
                p["g"][j, :] = trunc_grad(p["g"][j, :], vmax)
            
            p["v"][j, :] = get_velocity(
                x=p["x"][j, :],
                v=p["v"][j, :],
                p=p["p"][j, :],
                s=p["g"][j, :],
                gb=gbest,
                I=p["i"][j],
                params=params,
                direction=direction
            )

            p["v"][j, :] = trunc_vel(p["v"][j, :], vmax)

            p["x"][j, :] = p["x"][j, :] + p["v"][j, :]

            p["x"][j, :], p["i"][j], p["c"][i] = trunc_space(
                x=p["x"][j, :],
                i=p["i"][j],
                c=p["c"][j],
                r=rnge
            )

            p["xfit"][j] = cec17(x=p["x"][j, :], fun_nums=idf)

            p["x"][j, :], p["pfit"][j], g = update_best(
                x=p["x"][j, :],
                xfit=p["xfit"][j],
                p=p["p"][j, :],
                pfit=p["pfit"][j],
                g=gbest
            )

        p["i"], p["c"] = update_importance(
            x=p["x"],
            I=p["i"],
            fit=p["xfit"],
            oldfit=p["oldfit"],
            c=p["c"],
            g=p["g"],
            gx=g["x"],
            cmax=cmax,
            n=n
        )

        p["oldfit"] = p["xfit"]

        diversity = get_diversity(x=p["x"], l=l, n=n)

        direction, p["i"] = update_direction(
            direction=direction,
            diversity=diversity,
            I=p["i"],
            t=dt,
            n=n
        )

        if (g["xfit"] - g_opt(idf)) <= stopc:
            sr = 1
            break
    
    err = g["xfit"] - g_opt[idf]

    return err, sr, i
