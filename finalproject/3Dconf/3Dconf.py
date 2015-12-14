#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import symbols, sin, cos, Matrix, lambdify
from math import degrees as deg
import numpy as np
import itertools
import scipy.spatial.distance as dis
import sys


LANG = sys.stdout.encoding          # Get system language code
np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def transProc(pointFile, controlPtFile, outputFileName):
    """A function to control 3D conformal transformation process"""
    # Read 3D coordinates from file
    fin = open(controlPtFile)
    lines = fin.readlines()
    fin.close()

    data = np.array(map(lambda l: map(float, l.split()), lines))
    x, y, z, X, Y, Z = map(lambda e: e.flatten(), np.hsplit(data, 6))

    L = np.matrix(np.append(x, [y, z, X, Y, Z])).T

    # Define symbols
    Ss, Omegas, Phis, Kappas, Txs, Tys, Tzs = symbols(
        u"σ ω φ κ Tx Ty Tz".encode(LANG))
    dXs = np.array([Ss, Omegas, Phis, Kappas, Txs, Tys, Tzs])

    # Symbole for observations
    xs = np.array(symbols("x1:%d" % (len(x)+1)))
    ys = np.array(symbols("y1:%d" % (len(y)+1)))
    zs = np.array(symbols("z1:%d" % (len(z)+1)))
    Xs = np.array(symbols("X1:%d" % (len(X)+1)))
    Ys = np.array(symbols("Y1:%d" % (len(Y)+1)))
    Zs = np.array(symbols("Z1:%d" % (len(Z)+1)))
    Ls = np.array(np.append(xs, [ys, zs, Xs, Ys, Zs]))

    # Compute initial values
    S0, Omega0, Phi0, Kappa0, Tx0, Ty0, Tz0 = getInit(x, y, z, X, Y, Z)
    L0 = np.matrix(np.append(x, [y, z, X, Y, Z])).T

    # List observation equations
    F = getEqns(
        Ss, Omegas, Phis, Kappas, Txs, Tys, Tzs, xs, ys, zs, Xs, Ys, Zs)

    # Create lists for substitution of initial values and constants
    var = np.append([Ss, Omegas, Phis, Kappas, Txs, Tys, Tzs], Ls)

    # Compute cofficient matrix
    JFx = F.jacobian(dXs)
    JFl = F.jacobian(Ls)

    # Create function objects for two parts of cofficient matrix and f matrix
    FuncB = lambdify(tuple(var), JFx, modules='sympy')
    FuncA = lambdify(tuple(var), JFl, modules='sympy')
    FuncF = lambdify(tuple(var), F, modules='sympy')

    dX = np.ones(1)      # Initial value for iteration
    lc = 1              # Loop counter
    while abs(dX.sum()) > 10**-10:
        # Create lists of initial values and constants
        val = np.append([S0, Omega0, Phi0, Kappa0, Tx0, Ty0, Tz0], L0)

        # Substitute values for symbols
        B = np.matrix(FuncB(*val)).astype(np.double)
        A = np.matrix(FuncA(*val)).astype(np.double)
        F0 = np.matrix(FuncF(*val)).astype(np.double)
        F = -F0 - A * (L - L0)

        Qe = A * A.T
        We = Qe.I
        N = (B.T * We * B)                  # Compute normal matrix
        t = (B.T * We * F)                  # Compute t matrix
        dX = N.I * t                        # Compute the unknown parameters
        V = A.T * We * (F - B * dX)         # Compute residual vector

        # Update initial values
        S0 += dX[0, 0]
        Omega0 += dX[1, 0]
        Phi0 += dX[2, 0]
        Kappa0 += dX[3, 0]
        Tx0 += dX[4, 0]
        Ty0 += dX[5, 0]
        Tz0 += dX[6, 0]
        L0 = L + V

        # Output messages for iteration process
        print "Iteration count: %d" % lc, u"|ΔX| = %.6f" % abs(dX.sum())
        lc += 1         # Update Loop counter

    # Compute residual vector
    V = A.T * We * (F - B * dX)         # Compute residual vector

    # Compute error of unit weight
    s0 = ((V.T * V)[0, 0] / (B.shape[0] - B.shape[1]))**0.5

    # Compute other informations
    SigmaXX = s0**2 * N.I
    # SigmaVV = s0**2 * (A.T - A.T * We * B * N.I * B.T) \
    #    * (A.T * We - A.T * We * B * N.I * B.T * We).T
    param_std = np.sqrt(np.diag(SigmaXX))

    # Output results
    print "\nResiduals:"
    print ("%-8s"+" %-8s"*6) % (
        "Point", "x res", "y res", "z res", "X res", "Y res", "Z res")
    for i in range(0, len(V) / 6):
        print ("%-8d"+" %-8.4f"*6) % (i + 1, V[i, 0], V[i + 3, 0], V[i + 6, 0],
                                      V[i + 9, 0], V[i + 12, 0], V[i + 15, 0])

    print "\n3D comformal transformation parameters:"
    print ("%9s"+" %12s %12s") % ("Parameter", "Value", "Stan Err")
    print "%-10s %12.4f %12.5f" % ("Scale", S0, param_std[0])
    print "%-10s %12.4fd %11.5fd" % (
        "Omega(deg)", deg(Omega0), deg(param_std[1]))
    print "%-10s %12.4fd %11.5fd" % (
        "Phi(deg)", deg(Phi0), deg(param_std[2]))
    print "%-10s %12.4fd %11.5fd" % (
        "Kappa(deg)", deg(Kappa0), deg(param_std[3]))
    print "%-10s %12.4f %12.5f" % ("Tx", Tx0, param_std[4])
    print "%-10s %12.4f %12.5f" % ("Ty", Ty0, param_std[5])
    print "%-10s %12.4f %12.5f" % ("Tz", Tz0, param_std[6])

    print "\nStandard error of unit weight : %.4f" % s0
    print "Degree of freedom: %d" % (B.shape[0] - B.shape[1])

    # Transform 3D coordinates
    # Read 3D coordinates from file
    fin = open(pointFile)
    lines = fin.readlines()
    fin.close()

    data = np.array(map(lambda l: map(float, l.split()), lines))
    x, y, z = map(lambda e: e.flatten(), np.hsplit(data, 3))

    M = getM(Omega0, Phi0, Kappa0)

    X = S0 * (M[0, 0]*x + M[1, 0]*y + M[2, 0]*z) + Tx0
    Y = S0 * (M[0, 1]*x + M[1, 1]*y + M[2, 1]*z) + Ty0
    Z = S0 * (M[0, 2]*x + M[1, 2]*y + M[2, 2]*z) + Tz0

    # Write out results
    fout = open(outputFileName, "w")
    for i in range(len(X)):
        fout.write("%.8f %.8f %.8f\n" % (X[i], Y[i], Z[i]))
    fout.close()


def getM(Omega, Phi, Kappa):    # Compute rotation matrix M
    M = np.matrix([             # for omega, phi, kappa
        [
            cos(Phi)*cos(Kappa),
            sin(Omega)*sin(Phi)*cos(Kappa) + cos(Omega)*sin(Kappa),
            -cos(Omega)*sin(Phi)*cos(Kappa) + sin(Omega)*sin(Kappa)],
        [
            -cos(Phi)*sin(Kappa),
            -sin(Omega)*sin(Phi)*sin(Kappa) + cos(Omega)*cos(Kappa),
            cos(Omega)*sin(Phi)*sin(Kappa) + sin(Omega)*cos(Kappa)],
        [
            sin(Phi),
            -sin(Omega)*cos(Phi),
            cos(Omega)*cos(Phi)]
        ])

    return M


def getM2(Azimuth, Tilt, Swing):    # Compute rotation matrix M
    M = np.matrix([             # for azimuth, tilt and swing
        [
            -cos(Azimuth)*cos(Swing) - sin(Azimuth)*cos(Tilt)*sin(Swing),
            sin(Azimuth)*cos(Swing) - cos(Azimuth)*cos(Tilt)*sin(Swing),
            -sin(Tilt)*sin(Swing)],
        [
            cos(Azimuth)*sin(Swing) - sin(Azimuth)*cos(Tilt)*cos(Swing),
            -sin(Azimuth)*sin(Swing) - cos(Azimuth)*cos(Tilt)*cos(Swing),
            -sin(Tilt)*cos(Swing)],
        [
            -sin(Azimuth)*sin(Tilt),
            -cos(Azimuth)*sin(Tilt),
            cos(Tilt)]
        ])

    return M


def getInit(x, y, z, X, Y, Z):  # Compute initial values for 7 parameters
    # Compute rotation angles
    # Get three point combinations
    TRI = np.array(
        list(itertools.combinations(zip(X, Y, Z, range(len(X))), 3)))
    h = 0       # Initial value for finding the highest height of triangle
    index = 0   # A variable recording index of triangle having highest height
    for i in range(len(TRI)):
        dist = dis.pdist(TRI[i])
        a = dist.max()
        b, c = dist[dist != dist.max()]
        h1 = np.sqrt(b**2 - ((a**2+b**2-c**2)/(2*a))**2)
        if h < h1:
            h = h1
            index = i
    # A variable recording index of three points
    pt_index = TRI[index][:, -1].astype(int)

    # Points gaving strongest triangle
    tX = X[pt_index]
    tY = Y[pt_index]
    tZ = Z[pt_index]
    tx = x[pt_index]
    ty = y[pt_index]
    tz = z[pt_index]

    # Compute normal vector of triangle
    A = (tY[1]-tY[0])*(tZ[2]-tZ[0]) - (tY[2]-tY[0])*(tZ[1]-tZ[0])
    B = (tX[2]-tX[0])*(tZ[1]-tZ[0]) - (tX[1]-tX[0])*(tZ[2]-tZ[0])
    C = (tX[1]-tX[0])*(tY[2]-tY[0]) - (tX[2]-tX[0])*(tY[1]-tY[0])
    a = (ty[1]-ty[0])*(tz[2]-tz[0]) - (ty[2]-ty[0])*(tz[1]-tz[0])
    b = (tx[2]-tx[0])*(tz[1]-tz[0]) - (tx[1]-tx[0])*(tz[2]-tz[0])
    c = (tx[1]-tx[0])*(ty[2]-ty[0]) - (tx[2]-tx[0])*(ty[1]-ty[0])

    # Compute tilt and azimuth
    Ti = np.arctan2(C, np.sqrt(A**2 + B**2)) + (np.pi/2)
    Az = np.arctan2(A, B)
    ti = np.arctan2(c, np.sqrt(a**2 + b**2)) + (np.pi/2)
    az = np.arctan2(a, b)

    # Perform an initial rotation
    cM = getM2(Az, Ti, 0).astype(np.double)     # Control rotation matrix
    aM = getM2(az, ti, 0).astype(np.double)     # Arbitrary rotation matrix

    # Rotate two points in triangles
    X1 = cM[0, 0]*tX[:2] + cM[0, 1]*tY[:2] + cM[0, 2]*tZ[:2]
    Y1 = cM[1, 0]*tX[:2] + cM[1, 1]*tY[:2] + cM[1, 2]*tZ[:2]
    x1 = aM[0, 0]*tx[:2] + aM[0, 1]*ty[:2] + aM[0, 2]*tz[:2]
    y1 = aM[1, 0]*tx[:2] + aM[1, 1]*ty[:2] + aM[1, 2]*tz[:2]

    # Compute azimuth of two frame after rotated
    Az1 = np.arctan2((X1[1] - X1[0]), (Y1[1] - Y1[0]))
    az1 = np.arctan2((x1[1] - x1[0]), (y1[1] - y1[0]))

    # Compute swing
    sw = Az1 - az1

    # Compute final rotation matrix with azimuth, tilt and swing
    aM2 = getM2(az, ti, sw).astype(np.double)     # Arbitrary rotation matrix
    M = aM2.T * cM

    # Compute initial approximations
    Omega0 = np.arctan2(-M[2, 1], M[2, 2])
    Phi0 = np.arcsin(M[2, 0])
    Kappa0 = np.arctan2(-M[1, 0], M[0, 0])

    # Compute scale approximation
    S0 = (dis.pdist(zip(tX, tY, tZ))).mean() / \
        (dis.pdist(zip(tx, ty, tz))).mean()

    # Compute translations in x, y, z
    Tx0 = (tX - S0 * (M[0, 0]*tx + M[1, 0]*ty + M[2, 0]*tz)).min()
    Ty0 = (tY - S0 * (M[0, 1]*tx + M[1, 1]*ty + M[2, 1]*tz)).mean()
    Tz0 = (tZ - S0 * (M[0, 2]*tx + M[1, 2]*ty + M[2, 2]*tz)).mean()

    return S0, Omega0, Phi0, Kappa0, Tx0, Ty0, Tz0


def getEqns(S, Omega, Phi, Kappa, Tx, Ty, Tz, x, y, z, X, Y, Z):
    # Compute rotation matrix
    M = getM(Omega, Phi, Kappa)

    Fx = S * (M[0, 0]*x + M[1, 0]*y + M[2, 0]*z) + Tx - X
    Fy = S * (M[0, 1]*x + M[1, 1]*y + M[2, 1]*z) + Ty - Y
    Fz = S * (M[0, 2]*x + M[1, 2]*y + M[2, 2]*z) + Tz - Z

    return Matrix(np.append(Fx, [Fy, Fz]).flatten())


def main():
    transProc("model45.out", "common345.txt", "model45_34.out")

    return 0


if __name__ == '__main__':
    main()
