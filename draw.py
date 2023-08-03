#adapted from mandelbrot set sample code and https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array
import numpy as np
from numpy import pi
from colorsys import hls_to_rgb
import matplotlib.animation as animation
from numpy.random import default_rng
import matplotlib.pyplot as plt
import itertools

def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    #arg,r = np.log(r),arg+pi
    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.1)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c

def drawz(z):
    img = colorize(z)
    plt.imshow(img)
    plt.show()
