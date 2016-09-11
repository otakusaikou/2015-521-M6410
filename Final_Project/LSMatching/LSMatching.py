#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def getVal(img, x, y):
    """Return gray value from image without exceeding image extent"""
    # Ensure the coordinates of given point is in the image extent
    x = np.clip(x, 0, img.shape[1] - 1)
    y = np.clip(y, 0, img.shape[0] - 1)

    return img[y, x]


def getInterpolation(img, x, y):
    """Resample from right image, using bilinear interpolation"""
    # Generate array for the interpolated values
    values = np.zeros(len(x))

    # Filter out the points outside the image extent
    mask = (x >= 0) & ((x + 1) <= (img.shape[1] - 1)) & \
        (y >= 0) & ((y + 1) <= (img.shape[0] - 1))
    x = x[mask]
    y = y[mask]

    # Get coordinates of nearest four points
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = x0 + 1, y0 + 1

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

    # Update the value array
    values[mask] = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return values


def linearReg(leftImg, rightImg, coords):
    """Compute linear regression"""
    xl, yl, xr, yr = coords
    leftArr = (getVal(leftImg, xl, yl)).flatten()
    rightArr = (getVal(rightImg, xr, yr)).flatten()

    # Compute elements of covariance matrix
    Sx2 = (leftArr**2).sum() - leftArr.sum()**2 / len(leftArr)
    # Sy2 = (rightArr**2).sum() - rightArr.sum()**2 / len(rightArr)
    Sxy = (leftArr*rightArr).sum() \
        - (leftArr.sum()*rightArr.sum()/len(leftArr))

    beta = 1.0 * Sxy / Sx2      # The slope of regression line

    # Y intercept of regression line
    alpha = (rightArr.sum()/len(rightArr)) - (beta*leftArr.sum()/len(leftArr))

    return alpha, beta, leftArr


def lsMatching(leftPt, rightPt, windowSize, leftImg, rightImg):
    """Use least square matching to get subpixel level matching points
    LeftPt, RightPt (list): [row, col]
    windowSize (integer): size of search window
    leftImg, rightImg (numpy.array): source and target image"""
    # Define initial parameters for affine transformation
    a0 = float(rightPt[1] - leftPt[1])
    a1 = 1.0
    a2 = 0
    b0 = float(rightPt[0] - leftPt[0])
    b1 = 0
    b2 = 1.0

    # Get the image points coordinates within the window
    yi = range((leftPt[0]-windowSize/2), (1+leftPt[0]+windowSize/2))
    xi = range((leftPt[1]-windowSize/2), (1+leftPt[1]+windowSize/2))

    # For left image
    xl, yl = map(lambda e: e.flatten(), np.meshgrid(xi, yi))

    # For right image
    xr = (a0 + a1 * xl + a2 * yl).astype(int)
    yr = (b0 + b1 * xl + b2 * yl).astype(int)

    # Get iteration values for lsq-matching
    h0, h1, valA = linearReg(leftImg, rightImg, (xl, yl, xr, yr))

    X = np.ones(1)      # Initial value for iteration

    # Initial values for the second termination criteria
    delta = 1
    X0 = np.zeros(1)    # Variable for old unknown parameters
    lc = 1              # Loop counter
    fh0 = np.ones(windowSize**2)    # Coefficients of dh0 are constants
    while max(abs(X)) > 0.01 and delta > 0.05 and lc < 40:
        # Compute elemens of coefficient matrix and f matrix
        if lc > 1:
            # Update the image point coordinates in the right image
            xr = a0 + a1 * xl + a2 * yl
            yr = b0 + b1 * xl + b2 * yl

        # Point values in the right window
        valB = getInterpolation(rightImg, xr, yr)

        Bx = (getInterpolation(rightImg, xr + 1, yr) -
              getInterpolation(rightImg, xr - 1, yr)) / 2.0
        By = (getInterpolation(rightImg, xr, yr + 1) -
              getInterpolation(rightImg, xr, yr - 1)) / 2.0

        fa0 = h1 * Bx
        fa1 = xl * h1 * Bx
        fa2 = yl * h1 * Bx
        fb0 = h1 * By
        fb1 = xl * h1 * By
        fb2 = yl * h1 * By
        fh1 = valB
        f0 = (h0 + h1 * valB - valA).flatten()

        # Compute unknown parameters
        B = np.matrix(np.concatenate((
            fa0, fa1, fa2, fb0, fb1, fb2, fh0, fh1)).reshape(8, -1)).T

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
        delta = abs(max(abs(X)) - max(abs(X0)))
        X0 = X      # Update the old unknown parameters

        if abs(X).sum() > 200:
            print "Failed!"
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

    print xr, yr, Sigxr, Sigyr
    return leftPt[1], leftPt[0], xr, yr, Sigxr, Sigyr


def main():
    # Read conjugate points
    fin = open("SIFT_result.txt")
    lines = np.unique(np.array(fin.readlines()))
    data = np.array(map(lambda x: x.split(), lines)).astype(int)
    Lx, Ly, Rx, Ry = map(lambda x: x.flatten(), np.hsplit(data, 4))

    # Define size of subarray
    windowSize = 23

    # Read images
    leftImg = cv2.imread("01.jpg")
    rightImg = cv2.imread("02.jpg")

    # Convert image to gray scale
    leftGray = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY).astype(np.double)
    rightGray = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY).astype(np.double)

    # Least square matching
    Lx2, Ly2, Rx2, Ry2 = ([] for i in range(4))

    fout = open("result.txt", "w")
    for i in range(len(Lx)):
        result = lsMatching(
            [Ly[i], Lx[i]], [Ry[i], Rx[i]], windowSize, leftGray, rightGray)

        if result:
            Lx2.append(result[0])
            Ly2.append(result[1])
            Rx2.append(result[2])
            Ry2.append(result[3])
            fout.write("%d %d %.6f %.6f\n"
                       % (result[0], result[1], result[2], result[3]))
    fout.close()

    return 0


if __name__ == '__main__':
    main()
