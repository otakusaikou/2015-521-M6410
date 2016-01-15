#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def symDist(x0, y0, xp, yp, k1, k2, k3):
    """Get coordinates with symmetric radial lens distortion corrected."""
    # Compute distance from principal point to image point
    xbar = xp - x0
    ybar = yp - y0
    r = np.hypot(xbar, ybar)

    # Compute correct coordinates
    xc = xp + xbar * (k1 * r**2 + k2 * r**4 + k3 * r**6)
    yc = yp + ybar * (k1 * r**2 + k2 * r**4 + k3 * r**6)

    return xc, yc


def decDist(x0, y0, xp, yp, p1, p2):
    """Get coordinates with decentering distortion corrected."""
    # Compute distance from principal point to image point
    xbar = xp - x0
    ybar = yp - y0
    r = np.hypot(xbar, ybar)

    # Compute correct coordinates
    xc = xp + (p1 * (r**2 + 2 * xbar**2) + 2 * p2 * xbar * ybar)
    yc = yp + (2 * p1 * xbar * ybar + p2 * (r**2 + 2 * ybar**2))

    return xc, yc


def allDist(x0, y0, xp, yp, k1, k2, k3, p1, p2):
    """Get coordinates with both types of distortion corrected."""
    # Compute distance from principal point to image point
    xbar = xp - x0
    ybar = yp - y0
    r = np.hypot(xbar, ybar)

    # Compute distortion corrections
    xc = xp + (p1 * (r**2 + 2 * xbar**2) + 2 * p2 * xbar * ybar) + \
        xbar * (r**2 * k1 + r**4 * k2 + r**6 * k3)
    yc = yp + (2 * p1 * xbar * ybar + p2 * (r**2 + 2 * ybar**2)) + \
        ybar * (r**2 * k1 + r**4 * k2 + r**6 * k3)

    return xc, yc


def getDr(d, IO, type_):
    """Compute distortion with given radial distance."""
    # Define radial distance range
    xp = np.linspace(0, d, d*50)
    yp = 0

    # Compute distortion corrections
    x0 = IO["x0"]
    y0 = IO["y0"]
    xbar = xp - x0
    ybar = yp - y0
    r = np.hypot(xbar, ybar)

    if type_ == "symmetric":
        k1 = IO["k1"]
        k2 = IO["k2"]
        k3 = IO["k3"]
        dx = xbar * (r**2 * k1 + r**4 * k2 + r**6 * k3)
        dy = ybar * (r**2 * k1 + r**4 * k2 + r**6 * k3)
    elif type_ == "decentering":
        p1 = IO["p1"]
        p2 = IO["p2"]
        dx = (p1 * (r**2 + 2 * xbar**2) + 2 * p2 * xbar * ybar)
        dy = (2 * p1 * xbar * ybar + p2 * (r**2 + 2 * ybar**2))

    dr = np.hypot(dx, dy)

    return xp, dr


def drawDist(figName, pos, IO, type_, show_plt=False):
    # Compute distortion data
    # Define grid points
    x = np.arange(-IO["Fw"]/2+1, IO["Fw"]/2-1, ((IO["Fw"]/2-1)/6))
    y = np.arange(-IO["Fh"]/2+1, IO["Fh"]/2-1, ((IO["Fh"]/2-1)/6))
    xv, yv = map(lambda a: a.flatten(), np.meshgrid(x, y))

    # Build grid data for contour plot
    xi = np.linspace(-IO["Fw"]/2-1, IO["Fw"]/2+1, int(IO["Fw"] * 5)).flatten()
    yi = np.linspace(-IO["Fh"]/2-1, IO["Fh"]/2+1, int(IO["Fh"] * 5)).flatten()
    XI, YI = np.meshgrid(xi, yi)        # Get rectangular grid of xi and yi

    if type_ == "symmetric":
        title = "Symmetric radial distortion\n (focal length:%.4f, unit:mm)" \
            % IO["f"]                       # Define title for figure
        xc, yc = symDist(                   # Compute corrected coordinates
            IO["x0"], IO["y0"], xv, yv, IO["k1"], IO["k2"], IO["k3"])

        # Compute values for coutour plot
        Xc, Yc = map(lambda a: a.reshape(len(yi), len(xi)), symDist(
            IO["x0"], IO["y0"], XI.flatten(), YI.flatten(),
            IO["k1"], IO["k2"], IO["k3"]))

        ZI = np.hypot(Xc - XI, Yc - YI)

    elif type_ == "decentering":
        title = "Decentering distortion\n (focal length:%.4f, unit:mm)" \
            % IO["f"]
        xc, yc = decDist(
            IO["x0"], IO["y0"], xv, yv, IO["p1"], IO["p2"])

        # Compute values for coutour plot
        Xc, Yc = map(lambda a: a.reshape(len(yi), len(xi)), decDist(
            IO["x0"], IO["y0"], XI.flatten(), YI.flatten(),
            IO["p1"], IO["p2"]))

        ZI = np.hypot(Xc - XI, Yc - YI)
    elif type_ == "all":
        title = "All distortions\n (focal length:%.4f, unit:mm)" % IO["f"]
        xc, yc = allDist(
            IO["x0"], IO["y0"], xv, yv,
            IO["k1"], IO["k2"], IO["k3"], IO["p1"], IO["p2"])

        # Compute values for coutour plot
        Xc, Yc = map(lambda a: a.reshape(len(yi), len(xi)), allDist(
            IO["x0"], IO["y0"], XI.flatten(), YI.flatten(),
            IO["k1"], IO["k2"], IO["k3"], IO["p1"], IO["p2"]))

        ZI = np.hypot(Xc - XI, Yc - YI)

    # Create figure
    fig = plt.figure(figName, figsize=(9, 12), dpi=80)
    ax = fig.add_subplot(pos)

    # Set title
    plt.title(title, size=15)

    # Define x y axis range and set to equivalent scale
    plt.axis('equal')
    ax.axis([-IO["Fw"]/2 - 1, IO["Fw"]/2 + 1, -IO["Fh"]/2 - 1, IO["Fh"]/2 + 1])

    # Plot contour plot
    CS = plt.contour(XI, YI, ZI, cmap=plt.cm.brg)
    plt.clabel(CS, inline=1, fontsize=10)

    # Plot vectors of distortion
    plt.quiver(
        xv, yv, xc - xv, yc - yv, color='r', alpha=.5,
        angles='xy', scale_units='xy', scale=.01, width=.003)

    fig.tight_layout()                  # Adjust subplot layout

    # Show plot
    if show_plt is True:
        plt.show()


def drawFuncPlot(figName, pos, d, IO, type_, show_plt=False):
    # Compute distortion values
    x, dr = getDr(d, IO, type_)

    # Create figure
    fig = plt.figure(figName)
    ax = fig.add_subplot(pos)

    # Set title
    if type_ == "symmetric":
        plt.title(
            "Function of symmetric radial distortion\n to radial distance",
            size=15)
    elif type_ == "decentering":
        plt.title(
            "Function of decentering radial distortion\n to radial distance",
            size=15)

    # Set labels of two axis
    ax.set_xlabel("Radial distance(mm)", fontsize=12, rotation=0)
    ax.set_ylabel("Distortion ($\mu$m)", fontsize=12)

    # Set x axis interval and range
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.axis([0, d, 0, 180])

    plt.xlim([0, d + .2])               # Set x axis range
    plt.grid()                          # Enable grid line
    plt.plot(x, 1000 * dr, 'b-')        # Plot function
    fig.tight_layout()                  # Adjust subplot layout

    # Show plot
    if show_plt is True:
        plt.show()


def main():
    # Define interior orientation parameters
    IO = {
        "f": 31.742654,
        "x0": -.042970,
        "y0": .195158,
        "Fw": 22.749272,
        "Fh": 15.163800,
        "k1": .000178,
        "k2": -1.202 * 10**-7,
        "k3": .000000,
        "p1": 2.420 * 10**-5,
        "p2": -6.558 * 10**-6
        }

    # Draw distortion plot
    drawDist(0, 221, IO, "symmetric")
    drawDist(0, 222, IO, "decentering")
    # drawDist(0, 223, IO, "all")
    drawFuncPlot(0, 223, 10, IO, "symmetric")
    drawFuncPlot(0, 224, 10, IO, "decentering", True)

    return 0


if __name__ == '__main__':
    main()
