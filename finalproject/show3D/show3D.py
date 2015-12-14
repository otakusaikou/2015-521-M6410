#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    # Read exposure coordinates
    lines = open("model23.out").readlines()
    EO = np.array(map(lambda l: l.split(), lines[:2])).astype(np.double)
    X, Y, Z = np.hsplit(EO, 3)

    # Read object point coordinates
    Pts = np.array(map(lambda l: l.split(), lines[2:])).astype(np.double)
    XA, YA, ZA = np.hsplit(Pts, 3)

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Name x y label and set equal axis
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    # Plot 3D points
    ax.scatter(XA, YA, ZA, c="r")
    ax.scatter(X, Y, Z, c="b")

    plt.show()

    return 0

if __name__ == '__main__':
    main()
