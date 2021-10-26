import numpy
import math

gf = math.pow(2, 16) + 1

def div_mod(q, a, b): # a / b mod q
  y = 0
  for i in numpy.arange(1, b):
    y = ( q * i + a) / b
    if y == round(y):
      return y
  return y

def lagrange_poly(xi, fi, N):
  k = N - 1
  signo = math.pow(-1, k)

  numerador = numpy.ones(N, dtype = numpy.float64)

  for i in range(N):
    for j in range(N):
      if(i != j):
        numerador[i] = (numerador[i] * xi[j]) % gf
    numerador[i] = (numerador[i] * fi[i]) % gf
  
  denominador = numpy.ones(N, dtype = numpy.float64) 
  for i in range(N):
    for j in range(N):
      if(i != j):
        sub = (xi[i] - xi[j]) % gf
        denominador[i] = (denominador[i] * sub) % gf

  s = 0
  part = numpy.zeros(N, dtype=numpy.float64) 
  for i in range(N):
    #part[i] = div_mod(gf, numerador[i], denominador[i])
    part[i] = pow((int)(denominador[i]), -1, (int)(gf)) * numerador[i] % gf
    s = (s + part[i]) % gf
  s = (signo * s) % gf
  
  return s


# Descripcion: Resolver polynomio de Lagrange iterativamente para obtener todos los coeficientes
# xi: entrada de Polinomio de Lagrange
# fi: resultado de PolinomioLagrange 
# k: Orden de plinomio + 1 
# c: numero de coeficiemntes que desea recuperar
def lagrange_poly_iter(xi, fi, k, c):
  s = numpy.zeros(k, dtype = int) 
  s[0] = lagrange_poly(xi, fi, k)

  gi = fi
  
  for t in range(1, c):
    for i in range(k):
      #gi[i] = div_mod(gf, (gi[i]-s[t-1]) % gf, xi[i])
      gi[i] = pow((int)(xi[i]), -1, (int)(gf)) * ((gi[i]-s[t-1]) % gf) % gf
    s[t] = lagrange_poly(xi, gi, k-t+1)
  return s