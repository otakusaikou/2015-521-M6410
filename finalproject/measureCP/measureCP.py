#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pdb


flag = 0


def measureProc(image1, image2, image3, outputFileName):
    """A function to control measuring process"""
    fout = open(outputFileName, "w")

    # Read image
    leftImg = cv2.cvtColor(cv2.imread(image1), cv2.COLOR_BGR2RGB)
    middleImg = cv2.cvtColor(cv2.imread(image2), cv2.COLOR_BGR2RGB)
    rightImg = cv2.cvtColor(cv2.imread(image3), cv2.COLOR_BGR2RGB)

    # Create figure
    fig = plt.figure("Point Measuring Process")

    # Left image
    ax1 = fig.add_subplot(131)
    ax1.imshow(leftImg)

    # Middle image
    ax2 = fig.add_subplot(132)
    ax2.imshow(middleImg)

    # Right image
    ax3 = fig.add_subplot(133)
    ax3.imshow(rightImg)

    axList = [ax1, ax2, ax3]

    # Listen key press event
    fig.canvas.mpl_connect('key_press_event', lambda event: onKeyPress(event, fout, axList))

    plt.show()

    fout.close()


def onKeyPress(event, fout, axList):
    global flag
    if event.key == "alt+m":
        if event.xdata != None and event.ydata != None and event.inaxes == axList[flag]:
            event.inaxes.plot(float(event.xdata), float(event.ydata), "bo")
            plt.draw()
            print "Point measured!  Coordinates: (row=%.6f, col=%.6f)" % (float(event.ydata), float(event.xdata))
            if flag == 0 or flag == 1:
                fout.write("%.6f %.6f " % (float(event.xdata), float(event.ydata)))
                flag += 1
            else:
                fout.write("%.6f %.6f\n" % (float(event.xdata), float(event.ydata)))
                flag = 0


def main():
    measureProc("03.jpg", "04.jpg", "05.jpg", "ControlPt345.txt")

    return 0
   

if __name__ == '__main__':
    main()
