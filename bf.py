#!/usr/bin/env python
import sys
import time
import json
import math
import numpy as np
import numpy.linalg as la
from scipy.integrate import solve_ivp
from scipy import linalg
from numpy.linalg import solve
import itertools as it
import sys_func

# from sympy import *

logfilename = "__LOG__"


class DataStruct:
    def __init__(self):
        if len(sys.argv) != 2:
            print(f"Usage: python {sys.argv[0]} filename")
            sys.exit(0)
        fd = open(sys.argv[1], "r")
        self.dic = json.load(fd)
        fd.close()
        self.dim = len(self.dic["x0"])
        dim2 = self.dim**2
        self.dim3 = self.dim * int((self.dim + 1) / 2) * self.dim
        print(self.dim3)
        self.ve = self.dim + dim2
        self.vel = self.ve + self.dim
        self.ve2 = self.vel + self.dim3
        self.vel2 = self.ve2 + dim2
        if self.dic["btype"] == "G":
            self.mu = 1.0
        elif self.dic["btype"] == "I":
            self.mu = -1.0


def det_derivative(A, dA):
    dim = A.shape[0]
    ret = 0
    for i in range(dim):
        temp = A.copy()
        temp[:, i] = dA[:, i]
        ret += np.linalg.det(temp)
    return ret


def bialt_square(A):
    n = A.shape[0]
    bialt_dim = sum(range(n))
    result = np.zeros((bialt_dim, bialt_dim))
    temp = np.zeros((2, 2))
    result_idx = ((i, j) for i in range(bialt_dim) for j in range(bialt_dim))
    mul_idx = [(i, j) for i in range(1, n) for j in range(i)]
    for row, col in it.product(mul_idx, mul_idx):
        for i, j in it.product([0, 1], [0, 1]):
            temp[i, j] = A[row[i], col[j]]
        result[next(result_idx)] = np.linalg.det(temp)
    return result


def bialt_square_derivative(A, dA):
    n = A.shape[0]
    bialt_dim = sum(range(n))
    result = np.zeros((bialt_dim, bialt_dim))
    temp = np.zeros((2, 2))
    dtemp = np.zeros((2, 2))
    result_idx = ((i, j) for i in range(bialt_dim) for j in range(bialt_dim))
    mul_idx = [(i, j) for i in range(1, n) for j in range(i)]
    for row, col in it.product(mul_idx, mul_idx):
        for i, j in it.product([0, 1], [0, 1]):
            temp[i, j] = A[row[i], col[j]]
            dtemp[i, j] = dA[row[i], col[j]]
        result[next(result_idx)] = det_derivative(temp, dtemp)
    return result


def func(t, x, data):
    dim = data.dim  # number of dimension
    ind_ui, ind_uj = np.triu_indices(dim)

    p = np.zeros(4)
    for i in range(len(data.dic["params"])):
        p[i] = data.dic["params"][i]
    f, dfdx, dfdl, d2fdx2, d2fdxdl = sys_func.sys_func(t, x, p, data)

    dphidx = x[dim : data.ve].reshape(dim, dim).T
    dphidl = x[data.ve : data.vel]
    f.extend((dfdx @ dphidx).T.flatten())
    f.extend(dfdx @ dphidl + dfdl)

    v = x[data.vel : data.ve2].reshape(int(data.dim3 / dim), dim)
    X = np.zeros(dim**3).reshape(dim, dim, dim)
    X[ind_ui, ind_uj] = v  # make a symmetric tensor
    X[ind_uj, ind_ui] = v
    d2phidx2 = X.transpose(0, 2, 1)
    d2phidxdl = x[data.ve2 : data.vel2].reshape(dim, dim).T

    psi = (dfdx @ d2phidx2 + (d2fdx2 @ dphidx).T @ dphidx).transpose(0, 2, 1)
    # crop lower_triangular tensor
    f.extend(psi[ind_ui, ind_uj].flatten())
    f.extend(
        (
            dfdx @ d2phidxdl + ((d2fdx2 @ dphidx).T @ dphidl).T + (d2fdxdl @ dphidx)
        ).T.flatten()
    )
    return f


def fixed(data):
    dim = data.dim  # number of dimension
    fperiod = 2 * np.pi
    duration = data.dic["period"] * fperiod
    # tspan = np.arange(0, duration, data.dic['tick'])
    x0 = data.dic["x0"]
    vp = data.dic["variable_param"]
    count = 0
    while True:
        prev = x0
        phi = np.append(x0, np.eye(dim).flatten())
        phi = np.append(phi, np.zeros(dim))
        phi = np.append(phi, np.zeros(data.dim3))
        phi = np.append(phi, np.zeros(dim * dim))
        x = solve_ivp(
            func,
            (0, duration),
            phi,
            # method = 'DOP853',
            method="RK45",
            rtol=1e-8,
            args=(data,),
        )  # pass a singleton
        vec = x.y[:, -1]  # copy last column
        xs = vec[:dim]  # extract x
        dphidx = vec[dim : data.ve].reshape(dim, dim).T
        dphidl = vec[data.ve : data.vel]
        d2phidx2 = np.zeros(dim * dim * dim).reshape(dim, dim, dim)

        ind_ui, ind_uj = np.triu_indices(dim)
        v = vec[data.vel : data.ve2].reshape(int(data.dim3 / dim), dim)
        d2phidx2[ind_ui, ind_uj] = v  # make a symmetric tensor
        d2phidx2[ind_uj, ind_ui] = v
        d2phidx2 = d2phidx2.transpose(0, 2, 1)
        d2phidxdl = vec[data.ve2 : data.vel2].reshape(dim, dim).T

        if data.dic["btype"] != "NS":
            # G, PD version
            J = dphidx - data.mu * np.eye(dim)
            F = xs - prev
            F = np.append(F, la.det(J))  # la.det(J) is chi

            dchidx = np.array([det_derivative(J, d2phidx2[i]) for i in range(dim)])
            dchidx = np.append(dchidx, det_derivative(J, d2phidxdl))
        else:
            # NS version (Bialternate Product)
            J = bialt_square(dphidx) - np.eye(sum(range(dim)))
            F = xs - prev
            F = np.append(F, la.det(J))  # la.det(J) is chi

            dchidx = np.array(
                [
                    det_derivative(J, bialt_square_derivative(dphidx, d2phidx2[i]))
                    for i in range(dim)
                ]
            )
            dchidx = np.append(
                dchidx, det_derivative(J, bialt_square_derivative(dphidx, d2phidxdl))
            )

        dFdx = dphidx - np.eye(dim)
        dFdx = np.insert(dFdx, dFdx.shape[1], dphidl, axis=1)
        dFdx = np.insert(dFdx, dFdx.shape[0], dchidx, axis=0)

        h = solve(dFdx, -F)  # solve x from Ax = b

        delta = np.linalg.norm(h, ord=2)
        if delta < data.dic["errors"]:
            break
        if delta > data.dic["explosion"]:
            print("exploded.")
            sys.exit(-1)
        if count >= data.dic["ite_max"]:
            print(f"over {count} times iteration")
            sys.exit(-1)
        x0 = prev + h[:dim]
        data.dic["params"][vp] += h[-1]
        count += 1
    data.dphidx = dphidx
    data.dic["x0"] = x0
    return count


def main():
    data = DataStruct()
    with open(logfilename, mode="w") as logfile:
        while True:
            t_start = time.perf_counter()
            iteration = fixed(data)
            t_end = time.perf_counter()
            l = linalg.eig(data.dphidx.T)[0]
            comp = True if abs(l[0].imag) < 1e-8 else False
            str = "{0:2d} ".format(iteration)
            logstr = ""
            for i in range(len(data.dic["params"])):
                str += "{0: .5f} ".format(data.dic["params"][i])
                logstr += "{0:} ".format(data.dic["params"][i])
            for i in range(len(data.dic["x0"])):
                str += "{0: .8f} ".format(data.dic["x0"][i])
                logstr += "{0:} ".format(data.dic["x0"][i])
            if comp == True:
                str += "R "
                str += "{0: .7f} {1: .7f} ".format(l[0].real, l[1].real)
            else:
                str += "C "
                str += "{0: .7f} {1: .7f} ".format(l[0].real, l[0].imag)
                str += "ABS "
                str += "{0: .7f} ".format(abs(l[0]))
            str += "({0:}[ms])".format(int((t_end - t_start) * 1e03))
            # print(l)
            print(str)
            logfile.write(logstr + "\n")
            pos = data.dic["increment_param"]
            data.dic["params"][pos] += data.dic["dparams"][pos]


if __name__ == "__main__":
    main()
