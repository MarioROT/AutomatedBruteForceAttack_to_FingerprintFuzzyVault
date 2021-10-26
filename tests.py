# Pruebas de diferentes operaciones sobre las bóvedas y los archivos.

# from lagrange_matlab import lagrange_poly_iter
import numpy as np
from sympy import *
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def polev(coefs,rng):
    '''Incertar coeficientes del polinomio comenzando
    desde el valor independiente hacia la mayor potencia'''
    res = []
    if len(rng) == 2:
        xs = range(rng[0],rng[1]+1)
    else:
        xs = range
    print(type(rng))
    for i in xs:
        pr = 0
        # print(i)
        for p,j in enumerate(coefs):
            pr += j*i**p
            # print(j*i**p)
        res.append(pr)
    return res

def intercalar_mm(xs,y, valor):
  xs = np.array(xs)
  y = np.array(y)
  mayores = xs[xs > valor][:]
  menores = xs[xs[xs < valor].argsort()][::-1]
  mayoresy = y[np.where(xs>valor)]
  menoresy = y[np.where(xs<valor)][::-1]
  traslp = []
  traslpy = []
  i=0
  while i <= np.abs(len(mayores)-len(menores))+2 :
    while i < (min(len(menores),len(mayores))):
      traslp.append(mayores[i])
      traslp.append(menores[i])
      traslpy.append(mayoresy[i])
      traslpy.append(menoresy[i])
      i += 1
    if i*2 == len(xs):
      if len(traslp) == len(xs):
        return traslp, traslpy

    if len(mayores) ==  (min(len(menores),len(mayores))):
      traslp.append(menores[i])
      traslpy.append(menoresy[i])
      if len(traslp) == len(xs):
        return traslp, traslpy

    else:
      traslp.append(mayores[i])
      traslpy.append(mayoresy[i])
      if len(traslp) == len(xs):
        return traslp, traslpy
    i += 1
  return traslp, traslpy

def Lagrange(x,y,val=0,orden=0, mgraphs=False):
    '''Rubs, si quieres calcular un grado menor a dos, mejor calculalo a partir de 3 y tendrás los que necesite.'''
    xg, yg = x,y
    x,y = intercalar_mm(x,y,val)
    X = Symbol('X')
    if orden:
        orden
    else:
        orden = len(x)-1
    if mgraphs:
        fig, axs = plt.subplots(int(np.ceil(orden/2)),2, figsize=(15,15))
        fig.suptitle('Puntos ajustados distintos grados de polinomio')

    for k in range(1,orden+1):
        coef=[]
        Li,Fx=1,0
        for i in range(k+1):
            Li=1
            for j in range(k+1):
                if i != j:
                    Li*=(X-x[j])/(x[i]-x[j])
            Fx+=Li*y[i]
        for m in range(k+1):
            coef.append(simplify(collect(Fx,X)).coeff(X,m))
    #         print(coef)
        Fxe = Fx.evalf(subs={X:val})
        print('Polinomio de Lagrange de orden {}: {}'.format(k, simplify(Fx)))
        print('\t Resultado con el polinomio de Lagrange de orden {}:  {}'.format(k,Fxe))

        if mgraphs:
            poly = np.poly1d(coef[::-1])
            new_x = np.linspace(xg[0], xg[-1])
            new_y = poly(new_x)

            kr = k -1
            axs[int(np.floor(kr/2)), kr%2].plot(xg, yg, "kD", new_x, new_y,'c--')
            axs[int(np.floor(kr/2)), kr%2].set_title('Poliniomio de grado: {}'.format(k))

    gh = [float(i%(2**16)) for i in coef]
    print(gh)


# coeffs= [2,8,1,5,3,4,6]
# pointsX = [i for i in range(50,60)]
# pointsY = polev(coeffs,[50,59])
# # print(lagrange_poly_iter(pointsX, pointsY, 7, 7))
# print(pointsX)
# print(pointsY)
pointsX = [23092,11028,20912,48308,49723,15791,51551,40568,58045]
pointsY = [41347,1691,63227,44986,49986,9722,7764,4267,15494]
# print(lagrange_poly_iter(pointsX, pointsY, 9, 9))
# Lagrange(pointsX,pointsY)

cofs = [21234,56847,55004,58836,32331,4274,60882,19908,16469]
xss = [23092,11028,20912,48308,49723,15791,51551,40568,58045]
print(type(xss))
polev(cofs,xss)
import random
from itertools import combinations
random.shuffle(combinations(pointsX,2))
for i in pointsX:
    print(i)
for i in combinations(pointsX,2):
    print(i)
for i in sorted(combinations(pointsX,2), key=lambda k: random.random()):
    print(i)



import time

ght = [i for i in range(50000)]
ff = [i for i in range(50000*1000)]
# start_time = time.time()
start_time = time.time()
for i in range(len(ght)):
    ght[i] = ght[i]+1

print("--- %s seconds ---" % (time.time() - start_time))



import random
from itertools import combinations

points = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13.14),(15,16),(17,18),(19,20)]

for i in points
random.shuffle(combinations(pointsX,2))
for i in pointsX:
    print(i)
for i in combinations(pointsX,2):
    print(i)
for i in sorted(combinations(pointsX,2), key=lambda k: random.random()):
    print(i)

grg = random.randrange(0,50)
grg

import itertools, random
import numpy as np

r = 10
k = 8
kCombinations = list(itertools.combinations(r, k + 1))
rpm =  np.random.permutation(len(kCombinations))
for i in range(len(kCombinations)):
    ri = i
    i = rpm[i]
    subset = kCombinations[i]

import math as m
import random
def combs(n,r):
  return m.factorial(n)/(m.factorial(r)*m.factorial(n-r))

def readFile(textFile, method = 1):
    vault = []
    if method == 1:
        with open(textFile) as file:
            vault = [tuple(map(int,map(float, line.replace(' ','',2).split('   ')))) for line in file]
        return vault
    elif method == 2:
        with open(textFile) as file:
            vault = np.array([list(map(int,map(float, line.replace(' ','',2).split('   ')))) for line in file])
        return vault

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

FF = readFile('vault/PRU.txt')
FF
for i in range(int(combs(10,9))):
    print(random_combination(FF,9))



FF = readFile('Real_XYs/Real_XY108_7.txt')
ff = readFile('Pruebas/Vaults/Vault108_7.txt')
# ff = readFile('Pruebas/Reals/Real_XY108_7.txt')
count = 0
for i in ff:
    if i in FF:
        count +=1
print(count)


combs(72,9)
