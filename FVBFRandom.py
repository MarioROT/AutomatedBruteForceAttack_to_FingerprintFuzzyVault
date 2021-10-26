# Implementación automatizada sobre una serie de bóvedas con registro de resultados.
# Esta implementación si tiene agregado la iteración de las combinaciones
# bajo una seleccion aleatoria de distribución uniforme. Lo cual implica que
# es posible encontrar una combianción de puntos genuinos para vulnerar la bóveda
# de forma mas rápida siempre haya mas de N+1 puntos genuinos entre el total # DEBUG:
# puntos que se busca. 
import hashlib
from lagrange_matlab import lagrange_poly_iter
import itertools
import numpy as np
import random
import time
import pathlib
import math as m

def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """
    Regresa una lista de archivos en un directorio, dado como objeto de pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames

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

def combs(n,r):
  return m.factorial(n)/(m.factorial(r)*m.factorial(n-r))

def FVBrutForceAttack(VP, GP, N, save = False):
    VPName = VP
    VP = readFile(VP,1)
    GP = readFile(GP,2)
    if len(GP) <= N+1:
        print('\n No hay suficientes puntos genuinos para poder verificar el exito del ataque por fuerza bruta para la bóveda {}'.format(str(VPName)[-14:-4]))
        if save:
            # print('Entrada')
            lines = ['\n No hay suficientes puntos genuinos para poder verificar el exito del ataque por fuerza bruta para la bóveda {}'.format(str(VPName)[-14:-4])]
            with open(save, 'a') as f:
                f.writelines(lines)
        return GP
    xG = GP[:,0] # Puntos genuinos x
    yG = GP[:,1] # Puntos genuinos y

    F = lagrange_poly_iter(xG, yG, N+1, N+1) # Coeficientes del polinomio origianl
    hF = hashlib.sha256(F.__str__().encode('utf-8')).hexdigest() # valor hash del polinomio original (Este valor es conocido por el atacante)

    print("Iniciando ataque de fuerza bruta")

    for k in range(N, N+1): # calcula el polinomio de lagrange con k + 1 puntos tomados del vault con k = 2, 3, 4, 5, 6, 7, 8
        print("Probando combinaciones para k=", k)

        start_time = time.time()
        # kCombinations = list(itertools.combinations(VP, k + 1))
        # print(len(kCombinations))
        # rpm =  np.random.permutation(len(kCombinations))

        for i in range(int(combs(len(VP), k+1))):
            if i > 1000000:
                print('\n\nNo se ha encontrado el polinomio correcto en el primer millon de iteraciones para la bóveda: {}'.format(str(VPName)[-14:-4]))
                print('\tLas iteraciones promedio esperadas son: {}'.format(int(combs(len(VP), k+1))))
                if save:
                    # print('Entrada')
                    lines = ['\n\n No se ha encontrado el polinomio correcto en el primer millon de iteraciones para la bóveda: {}'.format(str(VPName)[-14:-4]), 'Las iteraciones promedio esperadas son: {}'.format(int(combs(len(VP), k+1)))]
                    with open(save, 'a') as f:
                        f.writelines(lines)
                return i
            ri = i
            subset = random_combination(VP,k+1)
            xi = []
            fi = []
            for t in subset:
                aux = list(t)
                xi.append(aux[0])
                fi.append(aux[1])
            fa = lagrange_poly_iter(xi, fi, k+1, k+1)
            hfa = hashlib.sha256(fa.__str__().encode('utf-8')).hexdigest()

            if ri % 2000 == 0:
                print('It -> R: {} Perm: {}   Pol: {}'.format(ri,i,fa))
            # if count == 1000:
            #     break
            if(hfa == hF):
                print("\n\n     #-------------------- Se encontro el polinomio correcto de la bóveda: {} --------------------#".format(str(VPName)[-14:-4]))
                print('\n\t It -> R: {} Perm: {}   Pol: {}'.format(ri,i,fa))
                print("\n\t\t\t\t\t--- Tomó {0:.3f} segundos ---".format(time.time() - start_time))
                if save:
                    # print('Entrada')
                    lines = ["\n\n     #-------------------- Se encontro el polinomio correcto de la bóveda: {} --------------------#".format(str(VPName)[-14:-4]),'\n\t It -> R: {} Perm: {}   Pol: {}'.format(ri,i,fa),"\n\t\t\t\t\t--- Tomó {0:.3f} segundos ---".format(time.time() - start_time)]
                    with open(save, 'a') as f:
                        f.writelines(lines)
                    # with open(save, 'w') as f:
                    #     f.write('\n'.join(lines))

                return fa, time.time() - start_time
                # exit()



# "--- The correct polynomial coefficients are: %d and it has taken %s seconds to find it ---" % (fa,time.time() - start_time)
#--------------- Ejecución iterativa del ataque por furza bruta sobre las huellas de la base de datos -----------#
import pathlib

Vaults = get_filenames_of_path(pathlib.Path('Vaults')) #'ExpOctubre/Vaults'))
RealPoints = get_filenames_of_path(pathlib.Path('Real_XYs')) #'ExpOctubre/Real_XYs'))#
N = 8

for i in range(len(Vaults)):
    FVBrutForceAttack(Vaults[i], RealPoints[i], N, save = 'Pruebas/ResultsComplete_17-10-21.txt')
