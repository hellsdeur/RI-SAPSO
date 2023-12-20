import numpy as np
import scipy.io

from cec17_functions import cec17_test_func
from functions import *

def main(
        # idf, t, d, stopc, gc
        ):
    # faux = 

    # x: Solution vector
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # nx: Number of dimensions
    nx = 10
    # mx: Number of objective functions
    mx = 1
    # func_num: Function number
    func_num = 1
    # Pointer for the calculated fitness
    f = [0]

    cec17_test_func(x, f, nx, mx, func_num)

    print(f[0])

    g = scipy.io.loadmat("g_opt.mat")
    g_opt = g["g_opt"]
    rnge = [-100, 100]

if __name__ == "__main__":
    main()