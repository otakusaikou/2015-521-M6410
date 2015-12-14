#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2


def match(fileName1, fileName2, threshold, show=False):
    """SIFT matching with opencv
    Reference : http://goo.gl/70Tk8G"""
    # Read image
    leftImg = cv2.imread(fileName1)
    rightImg = cv2.imread(fileName2)

    # Convert image to gray scale
    leftGray = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
    rightGray = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)

    # Create sift detector object
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(leftGray, None)
    kp2, des2 = sift.detectAndCompute(rightGray, None)

    # Create Brute-Force matching object with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)

    # Index of matching point in left image
    indexLeft = [e.queryIdx for e in good]

    # Index of matching point in right image
    indexRight = [e.trainIdx for e in good]

    # Get coordinates of matching points
    Lcol = np.array(map(lambda i: kp1[i].pt[0], indexLeft))
    Lrow = np.array(map(lambda i: kp1[i].pt[1], indexLeft))
    Rcol = np.array(map(lambda i: kp2[i].pt[0], indexRight))
    Rrow = np.array(map(lambda i: kp2[i].pt[1], indexRight))

    print "Number of matching points: %d" % len(Lcol)

    if show:
        # Create figure
        fig = plt.figure("SIFT matching result")

        # Left image
        ax1 = fig.add_subplot(121)
        ax1.imshow(leftImg, interpolation="none")
        ax1.plot(Lcol, Lrow, "bo")

        # Right image
        ax2 = fig.add_subplot(122)
        ax2.imshow(rightImg, interpolation="none")
        ax2.plot(Rcol, Rrow, "bo")

        plt.show()

    # Write out results
    fout = open("SIFT_result.txt", "w")
    fout.write("L-col L-row R-col R-row\n")
    for i in range(len(Lcol)):
        fout.write("%d %d %d %d\n" % (Lcol[i], Lrow[i], Rcol[i], Rrow[i]))
    fout.close()

    return Lcol, Lrow, Rcol, Rrow


def main():
    match("01.jpg", "02.jpg", 0.3, True)

    return 0

if __name__ == '__main__':
    main()
