#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import symbols, sin, cos, Matrix, lambdify
from math import degrees as deg
import numpy as np
import sys


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def getM(omega, phi, kappa):    # Compute rotation matrix M
    M = Matrix([
        [
            cos(phi)*cos(kappa),
            sin(omega)*sin(phi)*cos(kappa) + cos(omega)*sin(kappa),
            -cos(omega)*sin(phi)*cos(kappa) + sin(omega)*sin(kappa)],
        [
            -cos(phi)*sin(kappa),
            -sin(omega)*sin(phi)*sin(kappa) + cos(omega)*cos(kappa),
            cos(omega)*sin(phi)*sin(kappa) + sin(omega)*cos(kappa)],
        [
            sin(phi),
            -sin(omega)*cos(phi),
            cos(omega)*cos(phi)]
        ])

    return M


def getEqns(x0, y0, f, XL, YL, ZL, omega, phi, kappa, x, y, XA, YA, ZA):
    # Compute rotation matrix
    M = getM(omega, phi, kappa)

    # Compute shift between object point and exposure
    dX = XA - XL
    dY = YA - YL
    dZ = ZA - ZL

    # Compute terms in collinearity equation
    q = M[2, 0] * dX + M[2, 1] * dY + M[2, 2] * dZ
    r = M[0, 0] * dX + M[0, 1] * dY + M[0, 2] * dZ
    s = M[1, 0] * dX + M[1, 1] * dY + M[1, 2] * dZ

    # List collinearity equations
    Fx = f * (r / q) + x - x0
    Fy = f * (s / q) + y - y0

    return Matrix([Fx, Fy])


def relativeOrientation(Lx, Ly, Rx, Ry, f, x0, y0):
    XL = YL = OL = PL = KL = 0          # E.O. parameters for left photo
    ZL = f                              # O=Omega, P=Phi, K=Kappa

    XR = abs(Lx - Rx).mean()    # Part of E.O. parameters for right photo

    # Define initial values
    ZR0 = f
    YR0 = OR0 = PR0 = KR0 = 0

    # Define initial coordinate for object points
    X0 = XR * Lx / (Lx - Rx)
    Y0 = XR * Ly / (Lx - Rx)
    Z0 = f - (XR * f / (Lx - Rx))

    # Define symbols
    YRs, ZRs, ORs, PRs, KRs = symbols("YR ZR OR PR KR")
    xls, yls, xrs, yrs = symbols("xl yl xr yr")
    XAs, YAs, ZAs = symbols("XA YA ZA")

    # Define a symbol array for unknown parameters
    ls = np.array([ORs, PRs, KRs, YRs, ZRs, XAs, YAs, ZAs])

    # List observation equations
    F1 = getEqns(
        x0, y0, f, XL, YL, ZL, OL, PL, KL, xls, yls, XAs, YAs, ZAs)
    F2 = getEqns(
        x0, y0, f, XR, YRs, ZRs, ORs, PRs, KRs, xrs, yrs, XAs, YAs, ZAs)
    F = Matrix([F1, F2])

    # Create function objects for matrices BTB, BTf and F
    JFx = F.jacobian(ls)
    FuncBTB = lambdify(ls, (JFx.T*JFx), 'numpy')
    FuncBTf = lambdify(
                       np.append(ls, [xls, yls, xrs, yrs]),
                       (JFx.T*-F), 'numpy')

    FuncF0 = lambdify(np.append(ls, [xls, yls, xrs, yrs]), F, 'numpy')

    # Array for the observables and initial values
    XA = X0
    YA = Y0
    ZA = Z0
    OR = np.zeros(Lx.shape) + OR0
    PR = np.zeros(Lx.shape) + PR0
    KR = np.zeros(Lx.shape) + KR0
    YR = np.zeros(Lx.shape) + YR0
    ZR = np.zeros(Lx.shape) + ZR0
    l = np.dstack(
        ([OR, PR, KR, YR, ZR, XA, YA, ZA, Lx, Ly, Rx, Ry])).reshape(-1, 12)

    numPts = len(Lx)

    # Start iteration process
    lc = 1                      # Loop count
    C = 2.0                     # Stopping criteria
    VTV0 = 1.0                  # Initial value of residual
    while C > 10**-12:
        BTB = FuncBTB(*np.hsplit(l, 12)[:-4])
        BTf = FuncBTf(*np.hsplit(l, 12))

        N = np.matrix(np.zeros((5+3*numPts, 5+3*numPts)))
        t = np.matrix(np.zeros((5+3*numPts, 1)))

        N[:5, :5] = BTB[:5, :5].sum(axis=2).reshape(5, 5)
        t[:5, :] = BTf[:5, :].sum(axis=2).reshape(5, 1)         # Upper left
        for i in range(numPts):
            N[5+3*i:5+3*(i+1), :5] = \
                BTB[5:, :5, i].reshape(3, 5)
            t[5+3*i:5+3*(i+1), :] = \
                BTf[5:, :, i].reshape(3, 1)                     # Lower left

            for j in range(numPts):
                N[:5, 5+3*j:5+3*(j+1)] = \
                    BTB[:5, 5:, j].reshape(5, 3)                # Upper right
                if i == j:
                    N[5+3*i:5+3*(i+1), 5+3*j:5+3*(j+1)] = \
                        BTB[5:, 5:, i].reshape(3, 3)            # Lower right

        # Solve unknown parameters
        X = N.I * t

        # Compute residual
        F0 = np.matrix(FuncF0(*np.hsplit(l, 12))[:, 0, :, 0].T.reshape(-1, 1))
        VTV1 = X.T*N*X - t.T*X - X.T*t + F0.T*F0

        # Update the stopping criteria
        C = abs(VTV1/VTV0 - 1)
        VTV0 = VTV1

        # Update initial values
        l[:, 0] += X[0, 0]
        l[:, 1] += X[1, 0]
        l[:, 2] += X[2, 0]
        l[:, 3] += X[3, 0]
        l[:, 4] += X[4, 0]
        l[:, 5] += np.array(X[5::3, 0]).ravel()
        l[:, 6] += np.array(X[6::3, 0]).ravel()
        l[:, 7] += np.array(X[7::3, 0]).ravel()

        # Output messages for iteration process
        print "Iteration count: %d, " % lc, "VTV = %.6f" % VTV1

        lc += 1         # Update Loop counter

    # Compute sigma0
    s0 = (VTV1[0, 0] / (4*numPts - (5+3*(numPts))))**0.5

    # Compute other informations
    SigmaXX = s0**2 * N.I
    paramStd = np.sqrt(np.diag(SigmaXX))

    # Output results
    print "\nExterior orientation parameters:"
    print ("%9s"+" %9s"*3) % ("Parameter", "Left pho", "Right pho", "SD right")
    print "%-10s %8.4f %9.4f %9.4f" % (
        "Omega(deg)", deg(OL), deg(l[0, 0]), deg(paramStd[0]))
    print "%-10s %8.4f %9.4f %9.4f" % (
        "Phi(deg)", deg(PL), deg(l[0, 1]), deg(paramStd[1]))
    print "%-10s %8.4f %9.4f %9.4f" % (
        "Kappa(deg)", deg(KL), deg(l[0, 2]), deg(paramStd[2]))
    print "%-10s %8.4f %9.4f" % (
        "XL", XL, XR)
    print "%-10s %8.4f %9.4f %9.4f" % (
        "YL", YL, l[0, 3], paramStd[3])
    print "%-10s %8.4f %9.4f %9.4f\n" % (
        "ZL", ZL, l[0, 4], paramStd[4])

    print "Object point coordinates:"
    print ("%5s"+" %9s"*6) % ("Point", "X", "Y", "Z", "SD-X", "SD-Y", "SD-Z")

    for i in range(len(X0)):
        print ("P%d"+" %9.4f"*6) % (
            i, l[i, 5], l[i, 6], l[i, 7],
            paramStd[3*i+5],
            paramStd[3*i+6],
            paramStd[3*i+7])

    print "\nSigma0 : %.4f" % s0
    print "Degree of freedom: %d" % (4*numPts - (5+3*(numPts)))

    return l


def main():
    # Load I.O. parameters and image coordinate of conjugate points
    if len(sys.argv) == 1:      # Use test data if nothing given
        outputFileName = "result.txt"
        Lx = np.array([-4.87, 89.296, 0.256, 90.328, -4.673, 88.591])
        Ly = np.array([1.992, 2.706, 84.138, 83.854, -86.815, -85.269])
        Rx = np.array([-97.920, -1.485, -90.906, -1.568, -100.064, -0.973])
        Ry = np.array([-2.91, -1.836, 78.980, 79.482, -95.733, -94.312])
        f = 152.113     # Focal length in mm

    else:
        inputFileName, IOFileName, outputFileName = tuple(sys.argv[1:])
        imgPts = np.genfromtxt(inputFileName, skip_header=1)
        Lx, Ly, Rx, Ry = np.hsplit(imgPts, 4)
        f = np.genfromtxt(IOFileName)[0]

    x0 = y0 = 0

    # Compute the relative orientation parameters and object point coordinates
    RO = relativeOrientation(Lx, Ly, Rx, Ry, f, x0, y0)

    # Write out the results
    np.savetxt(
        outputFileName,
        RO[:, 5:8],
        fmt="%.8f %.8f %.8f",
        # header="X Y Z R G B",
        comments='')

    return 0


if __name__ == '__main__':
    main()
