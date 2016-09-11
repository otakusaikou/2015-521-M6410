#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import symbols, sin, cos, Matrix, lambdify, matrix2numpy
from math import degrees as deg
import numpy as np
import sys


# LANG = sys.stdout.encoding          # Get system language code
LANG = "utf-8"
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


def main(inputFileName, IOFileName, outputFileName):
    # Read image coordinates from file
    fin = open(inputFileName)
    lines = fin.readlines()
    fin.close()

    data = np.array(map(lambda l: map(float, l.split()), lines))
    Lx, Ly, Rx, Ry = map(lambda a: a.flatten(), np.hsplit(data, 4))

    # Read interior orientation information from file
    fin = open(IOFileName)
    data = map(lambda x: float(x), fin.readline().split())
    fin.close()

    f = data[0]

    # Define initial values
    XL0 = YL0 = OmegaL0 = PhiL0 = KappaL0 = 0
    ZL0 = ZR0 = f
    XR0 = abs(Lx - Rx).mean()
    YR0 = OmegaR0 = PhiR0 = KappaR0 = 0

    # Define initial coordinate for object points
    # X0 = np.array(Lx)
    # Y0 = np.array(Ly)
    # Z0 = np.zeros(len(Lx))
    X0 = XR0 * Lx / (Lx - Rx)
    Y0 = XR0 * Ly / (Lx - Rx)
    Z0 = f - ((XR0 * f) / (Lx - Rx))

    # Define symbols
    fs, x0s, y0s = symbols("f x0 y0")
    XLs, YLs, ZLs, OmegaLs, PhiLs, KappaLs = symbols(
        u"XL YL ZL ωL, φL, κL".encode(LANG))

    XRs, YRs, ZRs, OmegaRs, PhiRs, KappaRs = symbols(
        u"XR YR ZR ωR, φR, κR".encode(LANG))

    xls, yls, xrs, yrs = symbols("xl yl xr yr")
    XAs, YAs, ZAs = symbols("XA YA ZA")

    # List observation equations
    F1 = getEqns(
        x0s, y0s, fs, XLs, YLs, ZLs, OmegaLs, PhiLs, KappaLs,
        xls, yls, XAs, YAs, ZAs)

    F2 = getEqns(
        x0s, y0s, fs, XRs, YRs, ZRs, OmegaRs, PhiRs, KappaRs,
        xrs, yrs, XAs, YAs, ZAs)

    # Create lists for substitution of initial values and constants
    var1 = np.array([x0s, y0s, fs, XLs, YLs, ZLs, OmegaLs, PhiLs, KappaLs,
                     xls, yls, XAs, YAs, ZAs])
    var2 = np.array([x0s, y0s, fs, XRs, YRs, ZRs, OmegaRs, PhiRs, KappaRs,
                     xrs, yrs, XAs, YAs, ZAs])

    # Define a symbol array for unknown parameters
    l = np.array([OmegaRs, PhiRs, KappaRs, YRs, ZRs, XAs, YAs, ZAs])

    # Compute coefficient matrix
    JF1 = F1.jacobian(l)
    JF2 = F2.jacobian(l)

    # Create function objects for two parts of coefficient matrix and f matrix
    B1 = lambdify(tuple(var1), JF1, modules='sympy')
    B2 = lambdify(tuple(var2), JF2, modules='sympy')
    F01 = lambdify(tuple(var1), -F1, modules='sympy')
    F02 = lambdify(tuple(var2), -F2, modules='sympy')

    X = np.ones(1)      # Initial value for iteration
    lc = 1              # Loop counter
    while abs(X.sum()) > 10**-10:
        B = np.matrix(np.zeros((4 * len(Lx), 5 + 3 * len(Lx))))
        F0 = np.matrix(np.zeros((4 * len(Lx), 1)))
        # Column index which is used to update values of B and f matrix
        j = 0
        for i in range(len(Lx)):
            # Create lists of initial values and constants
            val1 = np.array([0, 0, f, XL0, YL0, ZL0, OmegaL0, PhiL0, KappaL0,
                             Lx[i], Ly[i], X0[i], Y0[i], Z0[i]])
            val2 = np.array([0, 0, f, XR0, YR0, ZR0, OmegaR0, PhiR0, KappaR0,
                             Rx[i], Ry[i], X0[i], Y0[i], Z0[i]])
            # For coefficient matrix B
            b1 = matrix2numpy(B1(*val1)).astype(np.double)
            b2 = matrix2numpy(B2(*val2)).astype(np.double)
            B[i*4:i*4+2, :5] = b1[:, :5]
            B[i*4:i*4+2, 5+j*3:5+(j+1)*3] = b1[:, 5:]
            B[i*4+2:i*4+4, :5] = b2[:, :5]
            B[i*4+2:i*4+4, 5+j*3:5+(j+1)*3] = b2[:, 5:]

            # For constant matrix f
            f01 = matrix2numpy(F01(*val1)).astype(np.double)
            f02 = matrix2numpy(F02(*val2)).astype(np.double)
            F0[i*4:i*4+2, :5] = f01
            F0[i*4+2:i*4+4, :5] = f02
            j += 1

        # Solve unknown parameters
        N = np.matrix(B.T * B)  # Compute normal matrix
        t = B.T * F0            # Compute t matrix
        X = N.I * t             # Compute the unknown parameters

        # Update initial values
        OmegaR0 += X[0, 0]
        PhiR0 += X[1, 0]
        KappaR0 += X[2, 0]
        YR0 += X[3, 0]
        ZR0 += X[4, 0]
        X0 += np.array(X[5::3, 0]).flatten()
        Y0 += np.array(X[6::3, 0]).flatten()
        Z0 += np.array(X[7::3, 0]).flatten()

        # Output messages for iteration process
        print "Iteration count: %d" % lc, u"|ΔX| = %.6f".encode(LANG) \
            % abs(X.sum())
        lc += 1         # Update Loop counter

    # Compute residual vector
    V = F0 - B * X

    # Compute error of unit weight
    s0 = ((V.T * V)[0, 0] / (B.shape[0] - B.shape[1]))**0.5

    # Compute other informations
    # Sigmall = np.eye(B.shape[0])
    SigmaXX = s0**2 * N.I
    # SigmaVV = s0**2 * (Sigmall - B * N.I * B.T)
    # Sigmallhat = s0**2 * (Sigmall - SigmaVV)
    param_std = np.sqrt(np.diag(SigmaXX))
    pho_res = np.array(V).flatten()
    # pho_res = np.sqrt(np.diag(SigmaVV))

    # Output results
    print "\nExterior orientation parameters:"
    print ("%9s"+" %9s"*3) % ("Parameter", "Left pho", "Right pho", "SD right")
    print "%-10s %8.4f %9.4f %9.4f" % (
        "Omega(deg)", deg(OmegaL0), deg(OmegaR0), deg(param_std[0]))
    print "%-10s %8.4f %9.4f %9.4f" % (
        "Phi(deg)", deg(PhiL0), deg(PhiR0), deg(param_std[1]))
    print "%-10s %8.4f %9.4f %9.4f" % (
        "Kappa(deg)", deg(KappaL0), deg(KappaR0), deg(param_std[2]))
    print "%-10s %8.4f %9.4f" % (
        "XL", XL0, XR0)
    print "%-10s %8.4f %9.4f %9.4f" % (
        "YL", YL0, YR0, param_std[3])
    print "%-10s %8.4f %9.4f %9.4f\n" % (
        "ZL", ZL0, ZR0, param_std[4])

    print "Object space coordinates:"
    print ("%5s"+" %9s"*6) % ("Point", "X", "Y", "Z", "SD-X", "SD-Y", "SD-Z")

    for i in range(len(X0)):
        print ("%5s"+" %9.4f"*6) % (
            ("p%d" % i), X0[i], Y0[i], Z0[i],
            param_std[3*i+5],
            param_std[3*i+6],
            param_std[3*i+7])

    print "\nPhoto coordinate residuals:"
    print ("%5s"+" %9s"*4) % ("Point", "xl-res", "yl-res", "xr-res", "yr-res")
    for i in range(len(X0)):
        print ("%5s"+" %9.4f"*4) % (
            ("p%d" % i), pho_res[2*i], pho_res[2*i+1],
            pho_res[2*i+len(X0)*2], pho_res[2*i+len(X0)*2+1])
    print ("\n%5s"+" %9.4f"*4+"\n") % (
        "RMS",
        np.sqrt((pho_res[0:len(X0)*2:2]**2).mean()),
        np.sqrt((pho_res[1:len(X0)*2+1:2]**2).mean()),
        np.sqrt((pho_res[len(X0)*2::2]**2).mean()),
        np.sqrt((pho_res[len(X0)*2+1::2]**2).mean()))

    print "Standard error of unit weight : %.4f" % s0
    print "Degree of freedom: %d" % (B.shape[0] - B.shape[1])

    # Write out results
    fout = open(outputFileName, "w")
    fout.write("%.8f %.8f %.8f\n" % (XL0, YL0, ZL0))
    fout.write("%.8f %.8f %.8f\n" % (XR0, YR0, ZR0))
    for i in range(len(X0)):
        fout.write("%.8f %.8f %.8f\n" % (X0[i], Y0[i], Z0[i]))
    fout.close()

    return 0


if __name__ == '__main__':
    inputFileName, IOFileName, outputFileName = tuple(sys.argv[1:])
    main(inputFileName, IOFileName, outputFileName)
