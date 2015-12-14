#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2


def showImgPair(fileName1, fileName2, pointFileName):
    """Display matching result of two images"""
    # Read image
    leftImg = cv2.imread(fileName1)
    rightImg = cv2.imread(fileName2)

    # Read conjugate points
    fin = open(pointFileName)
    lines = np.unique(np.array(fin.readlines()[1:]))
    data = np.array(map(lambda x: x.split(), lines)).astype(np.double)
    Lx, Ly, Rx, Ry = np.hsplit(data, 4)

    # Create figure
    fig = plt.figure("Matching Result")

    # Left image
    ax1 = fig.add_subplot(121)
    ax1.imshow(leftImg, interpolation="none")
    ax1.plot(Lx, Ly, "bo")

    # Right image
    ax2 = fig.add_subplot(122)
    ax2.imshow(rightImg, interpolation="none")
    ax2.plot(Rx, Ry, "bo")

    plt.show()


def main():
    showImgPair("01.jpg", "02.jpg", "SIFT_result.txt")

    return 0

if __name__ == '__main__':
    main()
