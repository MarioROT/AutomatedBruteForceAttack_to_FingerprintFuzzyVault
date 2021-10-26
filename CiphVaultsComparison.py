# Programa para verificar si mediante la lectura de una bóveda cifrada como archivo binario
# y transformado cada caracter a un numero entre 0 y 2^16 es posible obtener algunos de los
# puntos genuinos de cada bóveda, o los suficientes. Se guardan los resultados.

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

def readCiphVault(textFile, dec2_16= False):
    with open(textFile, 'rb') as binary_file:
        binary_file_data = binary_file.read()
        if dec2_16:
            file = [round((i/255)*2**16) for i in binary_file_data]
        else:
            file = binary_file_data
        return file

def combs(n,r):
    return m.factorial(n)/(m.factorial(r)*m.factorial(n-r))
def meancombs(n1,n2,r):
    return combs(n1,r)/combs(n2,r)

def IdentifyGenuinePoints(GP, CV, N, save = False):
    VaultName = str(CV)[-15:-5]
    GP = readFile(GP,2)
    CV = readCiphVault(CV, dec2_16 = True)
    GPL = GP[:,0] + GP[:,1] # Puntos genuinos en 'x' y 'y' concatenados

    count = 0

    for i in CV:
        if i in GPL:
            count += 1
    if count >= N+1:
        print('\n Sí se encontraron puntos genuinos suficientes para la bóveda: {}. Se encontraron {} puntos.\n\t Las combinaciones posibles {}.\n\t Las combinaciones promedio {}.'.format(VaultName, count, combs(len(CV),N+1), meancombs(len(CV),count,N+1)))
        if save:
            lines = ['\n Sí se encontraron puntos genuinos suficientes para la bóveda: {}. Se encontraron {} puntos.\n\t Las combinaciones posibles {}.\n\t Las combinaciones promedio {}.'.format(VaultName, count, combs(len(CV),N+1), meancombs(len(CV),count,N+1))]
            with open(save, 'a') as f:
                f.writelines(lines)
        return VaultName, count, combs(len(CV),N+1), meancombs(len(CV),count,N+1)
    print('\n No se encontraron puntos genuinos suficientes para la bóveda: {}. Se encontraron {} puntos'.format(VaultName, count))
    if save:
        lines = ['\n No se encontraron puntos genuinos suficientes para la bóveda: {}. Se encontraron {} puntos'.format(VaultName, count)]
        with open(save, 'a') as f:
            f.writelines(lines)
    return VaultName, count, 0, 0


# "--- The correct polynomial coefficients are: %d and it has taken %s seconds to find it ---" % (fa,time.time() - start_time)
#--------------- Ejecución iterativa del la revision de puntos generados de las bóvedas cifradas contra los puntos genuinos originales -----------#
import pandas as pd
import pathlib

CiphVaults = get_filenames_of_path(pathlib.Path('vault_cifrado')) #'ExpOctubre/Vaults'))
RealPoints = get_filenames_of_path(pathlib.Path('ExpOctubre/Real_XYs'))#
N = 8
vaultsL,countsL,combsL,meancombsL = [],[],[],[]
for i in range(len(CiphVaults)):
    VN,C,Combs,MeanCombs = IdentifyGenuinePoints(RealPoints[i], CiphVaults[i], N, save = 'ExpOctubre/CiphVaultsBFStats/VaultsCiphResultsComplete_26-10-21.txt')
    vaultsL.append(VN)
    countsL.append(C)
    combsL.append(Combs)
    meancombsL.append(MeanCombs)

df = pd.DataFrame({'Vaults': vaultsL,
                   'Count': countsL,
                   'Total Combinations': combsL,
                   'Mean Combinations': meancombsL})
df.to_csv('ExpOctubre/CiphVaultsBFStats/VaultsCiphResultsComplete_26-10-21.csv', index = False)
