from numpy import poly1d
import math

gf = math.pow(2, 16) + 1

def lagrange(x, w):
    M = len(x)
    p = poly1d(0.0)
    for j in range(M):
        pt = poly1d(w[j])
        for k in range(M):
            if k == j:
                continue
            fac = (x[j]-x[k]) % gf
            pt *= (poly1d([1.0, -x[k]])/fac) % gf
        p += pt
    return p
