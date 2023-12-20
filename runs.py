import numpy as np

from main import main

runs = 5
maxiter = 5000
npart = 20

dims = [30]
fname = "ackley"
rnge = [-47.5, 47.5]

r = np.zeros(shape=(runs, maxiter))
cr = np.zeros(shape=(1, runs))
it = np.zeros(shape=(1, runs))

# fe = np.zeros(shape=(1, runs))
ratio = 0

for j in range(len(dims)):
    for q in range(50):
        c = 0
        ratio = 0

        for i in range(runs):
            msg = [f"DIM: {dims[j]} | RUN: {i}"]
            print(msg)
            c = np.size(msg)

            r[i, :], it[i] = main(
                idf=maxiter,
                t=npart,
                d=dims[j],
                stopc=fname,
                gc=rnge
            )
            # fe[i] = get_global
            if r[i, -1] < 1e-10:
                ratio += 1
        
        print(f"DIM: {dims[j]}")
        print(f"Mean best: {r[:, -1]}")
        # print(f"FE: {np.mean(fe)}")
        print(f"Ratio: {ratio / runs * 100}")
        print(f"Iter: {np.mean(it)}")
        print('='*20)
