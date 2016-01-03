#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def pixel2fiducialProc(IOFileName, inputFileName, outputFileName):
    """A function to control transformation process"""
    # Read interior orientation information from file
    fin = open(IOFileName)
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

    # Read image point coordinates from file
    fin = open(inputFileName)
    data = np.array(map(lambda x: x.split(), fin.readlines()))
    fin.close()
    Lcol, Lrow, Rcol, Rrow = np.hsplit(np.double(data), 4)

    # Get point coordinates relative to fiducial axis
    Lxf, Lyf = img2frame(Lcol, Lrow, IO)
    Rxf, Ryf = img2frame(Rcol, Rrow, IO)

    # Compute corrected coordinates
    Lxc, Lyc = allDist(Lxf, Lyf, IO)
    Rxc, Ryc = allDist(Rxf, Ryf, IO)

    # Output results
    fout = open(outputFileName, "w")
    for i in range(len(Lxc)):
        fout.write("%.6f %.6f %.6f %.6f\n" % (Lxc[i], Lyc[i], Rxc[i], Ryc[i]))


def allDist(xp, yp, IO):
    """Get coordinates with both types of distortion corrected."""
    # Get coordinates of principal point relative to fiducial axis
    x0 = IO["xp"] - IO["Fw"]/2
    y0 = -(IO["yp"] - IO["Fh"]/2)

    # Compute distance from principal point to image point
    xbar = xp - x0
    ybar = yp - y0
    r = np.hypot(xbar, ybar)

    # Corrected coordinates with origin as principal point
    xc = xp - x0 + xbar * (r**2*IO["k1"]+r**4*IO["k2"]+r**6*IO["k3"]) + \
        (IO["p1"]*(r**2+2*xbar**2)+2*IO["p2"]*xbar*ybar)

    yc = yp - y0 + ybar * (r**2*IO["k1"]+r**4*IO["k2"]+r**6*IO["k3"]) + \
        (2*IO["p1"]*xbar*ybar+IO["p2"]*(r**2+2*ybar**2))

    return xc, yc


def img2frame(col, row, IO):
    "Transform row and column to fiducial coordinate system"
    Cp = col * IO["px"]
    Rp = row * IO["px"]

    return Cp - IO["Fw"]/2, -(Rp - IO["Fh"]/2)


def main():
    pixel2fiducialProc("IO.txt", "LSMatching12.txt", "fiducial12.txt")

    return 0

if __name__ == '__main__':
    main()
