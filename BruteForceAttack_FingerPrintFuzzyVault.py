# Implementación automatizada sobre una serie de bóvedas con registro de resultados.
#  Sin embargo esta implementación no tiene agregado la iteración de las combinaciones
# bajo una seleccion aleatoria de distribución uniforme. Lo cual implica que pueda llevar
# mucho tiempo para vulnerar una bóveda. La implementación que si cuenta con esta
# caracteristica se encuentra en el archivo --> FVBFRandom.py
import hashlib
from lagrange_matlab import lagrange_poly_iter
import itertools
import numpy as np
import random
import time
import pathlib


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


def FVBrutForceAttack(VP, GP, N, save = False):
    VPName = VP
    VP = readFile(VP,1)
    GP = readFile(GP,2)
    xG = GP[:,0] # Puntos genuinos x
    yG = GP[:,1] # Puntos genuinos y

    F = lagrange_poly_iter(xG, yG, N+1, N+1) # Coeficientes del polinomio origianl
    hF = hashlib.sha256(F.__str__().encode('utf-8')).hexdigest() # valor hash del polinomio original (Este valor es conocido por el atacante)

    print("Iniciando ataque de fuerza bruta")

    for k in range(N, N+1): # calcula el polinomio de lagrange con k + 1 puntos tomados del vault con k = 2, 3, 4, 5, 6, 7, 8
        print("Probando combinaciones para k=", k)

        start_time = time.time()
        kCombinations = list(itertools.combinations(VP, k + 1))
        print(len(kCombinations))
        rpm =  np.random.permutation(len(kCombinations))
        for i in range(len(kCombinations)):
            ri = i
            i = rpm[i]
            subset = kCombinations[i]
            xi = []
            fi = []
            for t in subset:
                aux = list(t)
                xi.append(aux[0])
                fi.append(aux[1])
            fa = lagrange_poly_iter(xi, fi, k+1, k+1)
            hfa = hashlib.sha256(fa.__str__().encode('utf-8')).hexdigest()

            if ri % 1000 == 0:
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
#--------------- Ejecución iterativa del ataque por fuerza bruta sobre las huellas de la base de datos -----------#
import pathlib

Vaults = get_filenames_of_path(pathlib.Path('Pruebas/Vaults'))
RealPoints = get_filenames_of_path(pathlib.Path('Pruebas/Reals'))
N = 8

for i in range(len(Vaults)):
    FVBrutForceAttack(Vaults[i], RealPoints[i], N, save = 'Pruebas/Results1.txt')
str(Vaults[1])[-14:-4]
