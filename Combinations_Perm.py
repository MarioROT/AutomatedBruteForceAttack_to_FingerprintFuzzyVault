# Programa para guardar en archivos distintas permutaciones de las
# combiaciones posibles para las bóvedas difusas
import numpy as np
import random
import itertools
import pathlib
import math as m

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

def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """
    Regresa una lista de archivos en un directorio, dado como objeto de pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames

def createPerms(FV,saveN,k=8,nperms=5, method = 0):
    FVName = FV
    p = pathlib.Path(str(FV)[-14:-4])
    path = saveN / p
    path.mkdir(parents=True, exist_ok=True)
    FV = readFile(FV)
    # print('Lenght FV: {}, calculo: {}'.format(len(FV), combs(len(FV),k+1)))
    kCombinations = list(itertools.combinations(FV, k + 1))
    for i in range(nperms):
        rpm =  np.random.permutation(len(kCombinations))
        # print('Lenght rpm: {}'.format(len(rpm)))
        if method == 0:
            PR = [str(kCombinations[idx]) for idx in rpm]
        else:
            PR = [str(idx) for idx in rpm]
        with open( path / (str(FVName)[-14:-4]+ 'Perm' + str(i) + '.txt'), 'a') as f:
            f.writelines('\n'.join(PR))
        f.close()



#-----------------------------------------------------------------------------#
# ejecición de las iteraciones.

import pathlib

Vaults = get_filenames_of_path(pathlib.Path('Pruebas/Vaults'))


for i in range(len(Vaults)):
    createPerms(Vaults[i],'Pruebas/Perms')

# import numpy as np
# rpm =  np.random.permutation(int(math.factorial(72)/(math.factorial(9)*math.factorial(72-9)))).dtype("uint8")
#
# import math
# (math.factorial(72)/(math.factorial(9)*math.factorial(72-9)))/1024**3
#
#
# import pathlib
# p = pathlib.Path('Hola'+"/")
# p.mkdir(parents=True, exist_ok=True)
# p
