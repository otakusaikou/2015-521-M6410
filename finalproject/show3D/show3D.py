#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from mpl_toolkits.mplot3d import Axes3D


def getEO(filename):
    """Read exterior orientation information from file"""
    fin = open(filename)
    data = np.array(map(
        lambda x: x.split(), fin.readlines())).astype(np.double)
    fin.close()

    X, Y, Z, Omega, Phi, Kappa = np.hsplit(data, 6)
    XYZ = np.append(X, [Y, Z])

    # Transform unit from degree to radians
    Omega = np.radians(Omega).flatten()
    Phi = np.radians(Phi).flatten()
    Kappa = np.radians(Kappa).flatten()

    return X, Y, Z, Omega, Phi, Kappa, XYZ.min(), XYZ.max()


def getIO(filename):
    """Read interior orientation information from file"""
    fin = open(filename)
    data = map(lambda x: float(x), fin.readline().split())
    fin.close()

    # Define interior orientation parameters
    IO = {
        "f": data[0],
        "xp": data[1],
        "yp": data[2],
        "Fw": data[3],
        "Fh": data[4],
        "px": data[5],
        "k1": data[6],
        "k2": data[7],
        "k3": data[8],
        "p1": data[9],
        "p2": data[10]}

    return IO


def getPt(filename):
    """Read object point coordinates from file"""
    fin = open(filename)
    data = np.array(map(
        lambda x: x.split(), fin.readlines())).astype(np.double)
    fin.close()
    X, Y, Z = np.hsplit(data, 3)

    return X, Y, Z, data.min(), data.max()


def getM(Omega, Phi, Kappa):
    """Compute rotation matrix M with omega, phi, kappa(rad)"""
    M = np.matrix([
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


def show3D(IOFileName, EOFilename, ptFilename, showEO=True, showPt=True):
    """Display position of object points and attitude of exposure stations.
    The two flags, 'showEO' and 'showPt' determine whether to show or hide the
    exposure stations and object points."""

    # Create 3D figure object
    fig = plt.figure("Result")
    ax = fig.add_subplot(111, projection='3d')

    # Name x, y and z label
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("Z axis")

    # Determine the bounds
    min_ = max_ = 0     # For overoll
    EOMin = EOMax = 0    # For exposure stations
    ptMin = ptMax = 0   # For object points

    # Plot exposure station
    if showEO:
        # Get interior and exterior orientation parameters
        Tx, Ty, Tz, Omega, Phi, Kappa, EOMin, EOMax = getEO(EOFilename)
        IO = getIO(IOFileName)

        # Get the extent of photo
        x0, y0 = 0 - IO["xp"], 0 - IO["yp"]
        x1, y1 = IO["Fw"] - IO["xp"], IO["Fh"] - IO["yp"]
        scale = .2      # Scale for photo extent

        # Coordinates of corner (Repeat first point for drawing rectangle)
        xp = np.array([x0, x1, x1, x0, x0]) * .2
        yp = np.array([y0, y0, y1, y1, y0]) * .2
        zp = -np.ones(5) * IO["f"] * scale
        for i in range(len(Tx)):
            M = getM(Omega[i], Phi[i], Kappa[i])
            X = (M[0, 0]*xp + M[1, 0]*yp + M[2, 0]*zp) + Tx[i]
            Y = (M[0, 1]*xp + M[1, 1]*yp + M[2, 1]*zp) + Ty[i]
            Z = (M[0, 2]*xp + M[1, 2]*yp + M[2, 2]*zp) + Tz[i]

            # Plot center of exposure station
            ax.scatter(Tx[i], Ty[i], Tz[i], c="r")

            # Plot photo extent
            ax.plot(X, Y, Z)

    # Plot object points
    if showPt:
        # Read object points
        XA, YA, ZA, ptMin, ptMax = getPt(ptFilename)
        ax.scatter(XA, YA, ZA, c="r")

    # Apply the bounds
    min_ = min(ptMin, EOMin)
    max_ = max(ptMax, EOMax)
    ax.auto_scale_xyz([min_, max_], [min_, max_], [min_, max_])

    plt.show()


def main():
    show3D("IO.txt", "EO.txt", "result.txt")

    return 0


if __name__ == '__main__':
    main()
