#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import datetime
from SIFTMatching import match
from LSMatching import lsMatchProc
from pixel2fiducial import pixel2fiducialProc


def addControlPt(fileName, targetFile1, targetFile2):
    """Write additional control information to results of LSM"""
    # Read common point coordinates of two models
    fin = open(fileName)
    lines = np.array(fin.readlines())
    data = np.array(map(lambda x: x.split(), lines)).astype(np.double)

    # L,M, R stand for left, middle and right image
    Lx, Ly, Mx, My, Rx, Ry = map(lambda x: x.flatten(), np.hsplit(data, 6))

    # Add control points to result of LSM
    fout1 = open(targetFile1, "a")
    fout2 = open(targetFile2, "a")
    for i in range(len(Lx)):
        fout1.write("%.6f %.6f %.6f %.6f\n" % (Lx[i], Ly[i], Mx[i], My[i]))
        fout2.write("%.6f %.6f %.6f %.6f\n" % (Mx[i], My[i], Rx[i], Ry[i]))
    fout1.close()
    fout2.close()


def main():
    """This is a script used to control the following programs:
        SIFTMatching.py: Use SIFT method to get matching points
        LSMatching.py: Use LSM as a filter to get more accurate result
        pixel2fiducial.py: Transform image coordinates to fiducial frame"""

    tStart = datetime.datetime.now()

    images = ["01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg"]
    threshold = [0.45, 0.36, 0.45, 0.5]
    IO = "IO.txt"
    commonCp = ["ControlPt123.txt", "ControlPt234.txt", "ControlPt345.txt"]

    for i in range(len(images) - 1):
        SIFTOutputFileName = "SIFT_Matching%d%d.txt" % (i + 1, i + 2)
        LSMOutputFileName1 = "LSMatching%d%d.txt" % (i, i + 1)
        LSMOutputFileName2 = "LSMatching%d%d.txt" % (i + 1, i + 2)
        imageCoordFileName = "fiducial%d%d.txt" % (i + 1, i + 2)

        match(
            images[i], images[i + 1],
            threshold[i], SIFTOutputFileName, False)

        lsMatchProc(
            images[i], images[i + 1],
            SIFTOutputFileName, LSMOutputFileName2)

        if i >= 1 and i <= len(commonCp):
            addControlPt(
                commonCp[i - 1], LSMOutputFileName1, LSMOutputFileName2)

    for i in range(len(images) - 1):
        LSMOutputFileName = "LSMatching%d%d.txt" % (i + 1, i + 2)
        imageCoordFileName = "fiducial%d%d.txt" % (i + 1, i + 2)
        pixel2fiducialProc(IO, LSMOutputFileName, imageCoordFileName)

    tEnd = datetime.datetime.now()
    print "Works done! It took %f sec" % (tEnd - tStart).total_seconds()

    return 0

if __name__ == '__main__':
    main()
