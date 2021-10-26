# Implementación sencilla para una sola bóveda a la vez del ataque por fuerza bruta
#  a un sistema de bóveda difusa basado en huella dactilar. Podria haber falta de
#  de memmoria RAM al momento de la ejección si los puntos son muchos y por tanto,
# segeneran muchas posibles combinaciones. Para solventarlo se descomentan
# las lineas 43 y 44, se comentna de la linea 45 a la 50. Aun que esto le quita la
# caracteristica de que las combinaciones se prueben de forma aleatoria bajo distribución uniforme.
import hashlib
from lagrange_matlab2 import lagrange
from lagrange_matlab import lagrange_poly_iter
import itertools
#from scipy.interpolate import lagrange
from lagrange_matlab2 import lagrange
import numpy as np
import random
import time

def readVault(textFile):  # lee el vault capturado en un archivo de texto
    vault = []
    with open(textFile) as file:
        vault = [tuple(map(float, line.split(' '))) for line in file]
    return vault

N = 8 # Grado del polinomio original
#Genuinos = [(27018, 39554),(20891, 49661),(42425, 26105),(15804, 21327),(21004, 49378),(53768, 17152),(9855, 59927),(30186, 42482),(47639, 31242),(29507, 18676),(7727, 35952),(50790, 64406),(39479, 63594),(27327, 25035),(18047, 1284),(37527, 53537),(32410, 10681),(24290, 22945),(36596, 25062),(4898, 13405),(14098, 18283),(20339, 64551),(58952, 38998),(59059, 18972),(59266, 6786)]

# xG = [2.4747000e+04,1.5791000e+04,2.9350000e+04,4.8308000e+04,2.8165000e+04,1.8052000e+04,5.3900000e+03,4.4727000e+04,1.1028000e+04] # Puntos genuinos x
# yG = [4.4870000e+04,9.7220000e+03,3.2492000e+04,4.4986000e+04,3.4110000e+04,2.8135000e+04,3.6404000e+04,3.7809000e+04,1.6910000e+03] # Puntos genuinos y

xG = [44663,28358,11892,36535,20914,59772,29964,50649,30859] # Puntos genuinos x
yG = [30150,2788,45338,1274,34488,47560,8083,26491,6286] # Puntos genuinos y

F = lagrange_poly_iter(xG, yG, N+1, N+1) # Coeficientes del polinomio origianl
hF = hashlib.sha256(F.__str__().encode('utf-8')).hexdigest() # valor hash del polinomio original (Este valor es conocido por el atacante)

r = readVault("vault/VP.txt")
#print(F)
print("Iniciando ataque de fuerza bruta")

for k in range(8, 9): # calcula el polinomio de lagrange con k + 1 puntos tomados del vault con k = 2, 3, 4, 5, 6, 7, 8
    print("Probando combinaciones para k=", k)

    start_time = time.time()
    # kCombinations = itertools.combinations(r, k + 1)
    # for subset in kCombinations:
    kCombinations = list(itertools.combinations(r, k + 1))
    rpm =  np.random.permutation(len(kCombinations))
    for i in range(len(kCombinations)):
        ri = i
        i = rpm[i]
        subset = kCombinations[i]
    # for count,subset in enumerate(kCombinations):
    # for subset in sorted(itertools.combinations(r, k + 1), key=lambda k: random.random()):
        xi = []
        fi = []
        for t in subset:
            aux = list(t)
            xi.append(aux[0])
            fi.append(aux[1])
        #print("xi = ", xi)
        #print("fi = ", fi)
        fa = lagrange_poly_iter(xi, fi, k+1, k+1)
        #fa = lagrange(xi, fi)
        hfa = hashlib.sha256(fa.__str__().encode('utf-8')).hexdigest()

        if ri % 1000 == 0:
            print('It -> R: {} Perm: {}   Pol: {}'.format(ri,i,fa))
        # print('It -> R: {}  Pol: {}'.format(count,fa))
        # if count == 1000:
        #     break
        if(hfa == hF):
            print("#--------- Se encontro el polinomio correcto:-----------#")
            # print(fa)
            print('It -> R: {} Perm: {}   Pol: {}'.format(ri,i,fa))
            print("--- %s seconds ---" % (time.time() - start_time))
            exit()
    print("--- %s seconds ---" % (time.time() - start_time))
