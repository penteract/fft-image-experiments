import numpy as np

def turn(n):
    """Formula from WIkipedia.
    n could be numpy array of integers
    """
    return (((n & -n) << 1) & n) != 0        

def dragon(N):
    """Generate dragon curve
    Returns a pair of integer arrays, (x,y), each 2^N elements long
    """
    t = turn(np.linspace(0, 2**N-1, 2**N, dtype=np.int32))

    a = np.remainder(np.cumsum(t*2-1), 4)

    #   1 | 0
    #   --+--  
    #   2 | 3
    dx = np.array([1, -1, -1, 1], dtype=np.int32)
    dy = np.array([1, 1, -1, -1], dtype=np.int32)
    
    
    x = np.cumsum(dx[a])
    y = np.cumsum(dy[a])

    return x-((dx-1)//2)[a],y-((dy-1)//2)[a]

def dragon_binary_diagram(N):
    """Draw dragon curve on a bitmap
    Returned bitmap size is 2^N x 2^N
    """
    #Prepare canvas to draw curve
    D = np.zeros((2**N,2**N), dtype=np.float32)

    #Get curve. Scale is 2x.
    dx, dy = dragon(2*N-1)

    dx *= 2
    dy *= 2

    #Center the curve.
    cx, cy = (int(dx.mean()), int(dy.mean()))
    x0 = cx - D.shape[0]//2
    y0 = cy - D.shape[1]//2
    dx -= x0
    dy -= y0

    #Given array of coordinates, writes 1 at theese coordinates, when they are inside canvas.
    def putOnesAt(dx,dy):
        inside = (dx >= 0) & (dx < D.shape[0]) & (dy>=0) & (dy<D.shape[0])
        #Take part of x,y coordinates that are inside the image, and write repeated pattern by them
        #
        D[dx[inside],dy[inside]] = 1

    #Draw corners
    putOnesAt(dx,dy)

    #Draw midpoints between corners
    dx1 = (dx[0:-1]+dx[1:])//2
    dy1 = (dy[0:-1]+dy[1:])//2
    putOnesAt(dx1,dy1)
    return D
    

def showdragon(N):
    pp.plot(*(dragon(N)+()))
    pp.show()

if __name__=="__main__":
    from matplotlib import pyplot as pp
    order = 16
    print("Showing dragon curve of order {}".format(order))
    showdragon(order)

def solid_dragon(N,k=0):
    """Here we are coloring edges"""
    l = 2**N
    D = np.zeros((l,l), dtype=np.float32)
    # D[h+x-y,h+x+y] is the edge in the +ve x direction from vertex x,y
    # D[h+x-y-1,h+x+y] is the edge in the +ve y direction from vertex x,y
    h = 2**(N-1) # starting point (vertex). D[o+x+y,] is the edge between either
    D[h,h]=1
    px=1
    py=1
    for i in range(2,N*2-k):
        # D = D + D[rotated 90 degrees clockwise about vertex ((px+py)/2,(py-px)/2]
        rot = D.copy()[::-1,::].transpose() # D rotated
        px,py=px-py,px+py # p = p - p rotated 90 degrees clockwise
        D += np.roll(rot, (px,py), (0,1))
    D+=D[::-1,::-1]
    #D+=D[::-1].transpose()
    return D

def part_dragon(N):
    D = np.zeros((2**N,2**N), dtype=np.float32)
    o = 2**(N-1)
    px=0
    py=1
    D[px::2,py::2] = 1
    def set_at_offset(dx,dy,rep):
        D[(dx+o)%rep::rep,(dy+o)%rep::rep]=1

    for i in range(2,N):
        px,py=px-py,px+py
        k=2**i
        set_at_offset(px,py,k)
        set_at_offset(-px,-py,k)
        px,py=px-py,px+py
        set_at_offset(px,py,k)
    return D
