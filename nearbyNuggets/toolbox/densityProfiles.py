import numpy as np


def sersicfunc(radius, amp, re, n):
#   calculate b
#    gg=special.gamma(2.0*n) # gamma function
#    bb=special.gammaincinv(2.0*n,gg/2) # invert the incomplete gamma function
    bb = 1.999*n - 0.327
    return amp*(np.exp(-1.0*bb*(radius/re)**(1/n)))
