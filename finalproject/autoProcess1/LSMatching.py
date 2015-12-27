#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def lsMatchProc(leftName, rightName, inputFileName, outputFileName):
    """A function to control LSM process"""
    # Read conjugate points from file
    fin = open(inputFileName)
    lines = np.unique(np.array(fin.readlines()))
    data = np.array(map(lambda x: x.split(), lines)).astype(int)
    Lx, Ly, Rx, Ry = map(lambda x: x.flatten(), np.hsplit(data, 4))

    # Define size of subarray
    windowSize = 15

    # Read images
    leftImg = cv2.imread(leftName)
    rightImg = cv2.imread(rightName)

    # Least square matching
    Lx2, Ly2, Rx2, Ry2 = ([] for i in range(4))

    # Counter for Least square matching points
    c = 0

    fout = open(outputFileName, "w")

    # Output processing message
    print "LSMatching process: ",
    msg = " " * (len(str(len(Lx))) - len(str(c))) + ("%d/%d" % (c, len(Lx)))
    sys.stdout.write(msg)

    for i in range(len(Lx)):
        result = lsMatching(
            [Ly[i], Lx[i]], [Ry[i], Rx[i]], windowSize, leftImg, rightImg)

        if result:
            Lx2.append(result[0])
            Ly2.append(result[1])
            Rx2.append(result[2])
            Ry2.append(result[3])
            fout.write("%d %d %.6f %.6f\n"
                       % (result[0], result[1], result[2], result[3]))

        # Output processing message
        sys.stdout.write("\b" * len(msg))
        c += 1
        msg = " " * (len(str(len(Lx))) - len(str(c))) \
            + ("%d/%d" % (c, len(Lx)))
        sys.stdout.write(msg)

    print
    fout.close()


def getRVal(img, x, y):
    """ Resample from right image, using bilinear interpolation"""
    # Get coordinates of nearest four points
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Ensure the coordinates of four points are in the right image extent
    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    # Get intensity of nearest four points
    Ia = img[y0, x0]  # Upper left corner
    Ib = img[y1, x0]  # Lower left corner
    Ic = img[y0, x1]  # Upper right corner
    Id = img[y1, x1]  # Lower right corner

    # Compute the weight of four points
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def linearReg(
        leftImg, rightImg, leftPt, rightPt, windowSize,
        a0, a1, a2, b0, b1, b2):
    """Compute linear regression"""
    leftArr = []    # Create two subarrays for left and right image
    rightArr = []
    for yl in range((leftPt[0]-windowSize/2), (1+leftPt[0]+windowSize/2)):
        for xl in range((leftPt[1]-windowSize/2), (1+leftPt[1]+windowSize/2)):
            xr = a0 + a1 * xl + a2 * yl     # Get corresponding right image
            yr = b0 + b1 * xl + b2 * yl     # coordinates
            leftArr.append(leftImg[yl, xl])
            rightArr.append(rightImg[yr, xr])

    leftArr = np.array(leftArr)
    rightArr = np.array(rightArr)

    # Compute elements of covariance matrix
    Sx2 = (leftArr**2).sum() - leftArr.sum()**2 / len(leftArr)
    # Sy2 = (rightArr**2).sum() - rightArr.sum()**2 / len(rightArr)
    Sxy = (leftArr*rightArr).sum() \
        - (leftArr.sum()*rightArr.sum()/len(leftArr))

    beta = 1.0 * Sxy / Sx2      # The slope of regression line

    # Y intercept of regression line
    alpha = (rightArr.sum()/len(rightArr)) - (beta*leftArr.sum()/len(leftArr))

    return alpha, beta


def lsMatching(leftPt, rightPt, windowSize, leftImg, rightImg):
    """Use least square matching to get subpixel level matching points
    LeftPt, RightPt (list): [row, col]
    windowSize (integer): size of search window
    leftImg, rightImg (numpy.array): source and target image"""
    # Convert image to gray scale
    leftGray = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY).astype(np.double)
    rightGray = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY).astype(np.double)

    # Define initial parameters for affine transformation
    a0 = float(rightPt[1] - leftPt[1])
    a1 = 1.0
    a2 = 0
    b0 = float(rightPt[0] - leftPt[0])
    b1 = 0
    b2 = 1.0

    # Get iteration values for lsq-matching
    h0, h1 = linearReg(leftGray, rightGray, leftPt, rightPt, windowSize,
                       a0, a1, a2, b0, b1, b2)

    X = np.ones(1)      # Initial value for iteration

    # Initial values for the second termination criteria
    delta = 1
    X0 = np.zeros(1)    # Variable for old unknown parameters
    lc = 1              # Loop counter

    while max(X) > 0.01 and delta > 0.05 and lc < 40:
        # Create lists for elements of coefficient matrix
        fa0, fa1, fa2, fb0, fb1, fb2, fh1, f0 = ([] for i in range(8))
        fh0 = np.ones(windowSize**2)    # Coefficients of dh0 are constants

        # Compute elemens of coefficient matrix and f matrix
        for yl in range((leftPt[0]-windowSize/2), (1+leftPt[0]+windowSize/2)):
            for xl in range(
                    (leftPt[1]-windowSize/2), (1+leftPt[1]+windowSize/2)):
                xr = a0 + a1 * xl + a2 * yl     # Get corresponding right image
                yr = b0 + b1 * xl + b2 * yl     # coordinates

                # Compute slope of B in both x and y directions
                Bx = (getRVal(rightGray, xr + 1, yr)
                      - getRVal(rightGray, xr - 1, yr)) / 2.0
                By = (getRVal(rightGray, xr, yr + 1)
                      - getRVal(rightGray, xr, yr - 1)) / 2.0

                fa0.append(h1 * Bx)
                fa1.append(xl * h1 * Bx)
                fa2.append(yl * h1 * Bx)
                fb0.append(h1 * By)
                fb1.append(xl * h1 * By)
                fb2.append(yl * h1 * By)
                fh1.append(getRVal(rightGray, xr, yr))
                f0.append(
                    h0 + h1 * getRVal(rightGray, xr, yr) - leftGray[yl, xl])

        # Compute unknown parameters
        B = np.matrix(zip(fa0, fa1, fa2, fb0, fb1, fb2, fh0, fh1))
        F = -np.matrix(f0).T

        N = B.T * B     # Compute normal matrix
        t = B.T * F     # Compute t matrix
        X = N.I * t     # Compute the unknown parameters

        # Update initial values
        a0 += X[0, 0]
        a1 += X[1, 0]
        a2 += X[2, 0]
        b0 += X[3, 0]
        b1 += X[4, 0]
        b2 += X[5, 0]
        h0 += X[6, 0]
        h1 += X[7, 0]

        # Compute the difference between the new and old unknown parameters
        delta = abs(max(X) - max(X0))
        X0 = X      # Update the old unknown parameters

        if abs(X).sum() > 200:
            # print "Failed!"
            return None

        lc += 1

    V = B * X - F   # Compute residual vector

    # Result of least square matching
    xr = a0 + a1 * leftPt[1] + a2 * leftPt[0]
    yr = b0 + b1 * leftPt[1] + b2 * leftPt[0]

    # Compute error of unit weight
    s0 = np.sqrt(((V.T*V) / (B.shape[0] - B.shape[1])))[0, 0]

    # Compute error of matching point
    JFx = np.matrix([
        [1, leftPt[1], leftPt[0], 0, 0, 0, 0, 0],
        [0, 0, 0, 1, leftPt[1], leftPt[0], 0, 0]])
    SigmaXX = s0 * N.I
    Sigxr, Sigyr = np.sqrt(np.diag(JFx * SigmaXX * JFx.T))

    # print xr, yr, Sigxr, Sigyr
    return leftPt[1], leftPt[0], xr, yr, Sigxr, Sigyr


def main():
    lsMatchProc("01.jpg", "02.jpg", "SIFT_Matching12.txt", "LSMatching12.txt")

    return 0


if __name__ == '__main__':
    main()
