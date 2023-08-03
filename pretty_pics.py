from math import pi, log, sqrt
from numpy import array, hstack, vstack, clip, histogram
from numpy.fft import fft2,fftshift
from PIL import Image
import numpy as np
from colorsys import hls_to_rgb

def blur(a):
    kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

def value_diapason(x, percent=0.95, nbins=100):
    """Use histogram to determine interval, covering 95% of values"""
    counts, bins = histogram(x.ravel(),nbins)
    total = sum(counts)
    accum = 0
    low = bins[-1]
    high = bins[0]
    #enumerate histogram bins starting from the most populated. 
    for i, cnt in sorted(enumerate(counts), 
                          key = (lambda i_c: i_c[1]),
                          reverse=True):
        accum += cnt
        low = min(low, bins[i])
        high = max(high, bins[i+1])
        if accum > percent * total:
            break
    return low, high
    

def toimage(fimg, gamma=1., percent=0.95, extend = 1.1, save=None):
    """Show binary matrix as monochrome image, automatically detecting upper and lower brightness bounds
    """
    low, high = value_diapason(fimg, percent=percent)
    print(low,high)

    mid = (low+high)/2
    low = mid + (low-mid)*extend
    high = mid + (high-mid)*extend
    low=max(low,fimg.min())
    high=min(high,fimg.max())
    
    image = Image.fromarray((clip((fimg-low)/(high-low),0,1)**gamma*255).astype(np.uint8), "P")
    if save is not None:
        image.save(save)
        print("Saving image file: {}".format(save))
    return image

def toimagecol(fimg, save=None):
    """save a grid of complex numbers as an image file"""
    colorize(fimg)
    image = Image.fromarray(colorize(fimg).astype(np.uint8), "RGB")
    if save is not None:
        image.save(save)
        print("Saving image file: {}".format(save))
    return image

def colorize(z):
    """colors a grid of complex numbers so that hue indicates argument and brightness indicates modulus"""
    r = np.abs(z)
    high = np.percentile(r,97.5)
    r = clip(r/high,0,1)**1
    arg = np.angle(z)

    #arg,r = np.log(r),arg+pi
    h = (arg )  / (2 * pi)
    l = (r)/2 # 1.0 - 1.0/(1.0 + r**0.1)
    #l=r/2
    s = 0.5

    c = np.vectorize(hls_to_rgb) (h,l*255,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c

def fourimage(a,b,c,d,save=None):
    cors = []
    for im in [a,b,c,d]:
        cors.append(colorize(im))
    r1 = np.concatenate((cors[0],cors[1]))
    r2 = np.concatenate((cors[2],cors[3]))
    r = np.concatenate((r1,r2),axis=1).astype(np.uint8)
    image = Image.fromarray(r, "RGB")

    if save is not None:
        image.save(save)
        print("Saving image file: {}".format(save))
    return image

def takeHalfandRotate(pattern,parity=1):
    """Given a square grid tiling the plane,
    takes pixels with odd coordinates and return a square grid the same size
    containing those points rotatad 45 degrees about the top left corner
    again representing a tiling"""
    sh = pattern.shape
    a,b = np.indices(sh)
    #a-=1
    return pattern[(a-b)%sh[0],(a+b+parity)%sh[1]]

if __name__=="__main__":
    N = 13
    from dragon import solid_dragon
    D = solid_dragon(N,0)#toimage(D).show()
    toimage(D, save="hi_res_backgrounds/dragon_{}.png".format(N))
    f = fft2(D)
    fimg = np.abs(f)#(np.abs(f))#np.log(np.abs(f)+1e-100)#np.real(f)
    #f2 = fimg**0.5 #  / blur(fimg)**0.5
    #fimg[1::2,0::2]*=-1
    #fimg[0::2,1::2]*=-1
    D2 = fft2(fimg)
    #fourimage(D,fftshift(f),fftshift(fimg),D2,save="dragon2_{}_quad.png".format(N))
    toimage(np.abs(D2), save="hi_res_backgrounds/dragon_fft_abs_fft_{}.png".format(N))
    #toimage((np.abs(D2)), save="dragon_diagram2_fft_fft{}.png".format(N))
    #toimage(fftshift(np.abs(fimg)), save="dragon_diagram2_{}_fft.png".format(N))
    #f2 = takeHalfandRotate(f)
    sh = D.shape
    x,y = np.indices(sh)
    k = np.exp(1j*pi*(x+y)/sh[0])
    toimagecol(fft2(f*(1+1j)/k),"hi_res_backgrounds/dragon_fft_derainbow_fft_{}.png".format(N))
    #fourimage(f2,takeHalfandRotate(fft2(f2),0),f,fft2(f*(1+1j)/k),save="dragon2_{}_quad2.png".format(N))
