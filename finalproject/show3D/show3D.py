#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    expNum = 5   # Define number of exposure station

    # Read exposure coordinates from file
    fin = open("result.txt")
    lines = fin.readlines()
    fin.close()

    data = np.array(map(lambda l: l.split(), lines)).astype(np.double)
    X, Y, Z = np.hsplit(data, 3)

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Name x, y and z label and set equal axis
    ax.pbaspect = [1.0, 1.0, 1.0]
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    # Plot 3D points
    # Exposure station
    ax.scatter(X[:expNum], Y[:expNum], Z[:expNum], c="b", marker="x")

    # Object point
    ax.scatter(X[expNum:], Y[expNum:], Z[expNum:], c="r", marker="o")

    plt.show()

    return 0

if __name__ == '__main__':
    main()
