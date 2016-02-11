#Read surprisal stuff from folder of matrices

import numpy as np
from scipy.sparse import lil_matrix
from surprisal import SurprisalAnalysis
from scipy.io import mmread, mmwrite
from samplecounts import getCounts
import matplotlib.pyplot as plt
import math

def getUncertainty(var, weights):
        "Get JSD uncertainty matrix"
        num = len(var)
        mat = np.zeros(num)
        for i in range(num):
                mat[i] = var[i] * (weights[i]**2)
        var2 = np.sum(mat)
        var2 = math.sqrt(var2)
        mat = np.sqrt(mat)
        return mat, var2

i = 10000
#Provide path to matrices here
add1 = './Round3ff96/Jobs/job1/matrix1/mat750.npy'
add2 = './Round3ff96/Jobs/job1/matrix2/mat750.npy'
matone = np.load(add1)
mattwo = np.load(add2)

matrixone = lil_matrix(matone)
matrixtwo = lil_matrix(mattwo)
commat = './tCounts.mtx'
#####Generate pseudocounts here. Maybe this isn't necessary but I wanted to stay consistent.
#omatrixone = lil_matrix(getCounts(commat, 150))
#omatrixtwo = lil_matrix(getCounts(commat, 150))
#np.set_printoptions(threshold = np.nan)
#matrixone = matrixone + omatrixone
#matrixtwo = matrixtwo + omatrixtwo
matrices = []
#Create list of matrices for surprisal
matrices.append(matrixone)
matrices.append(matrixtwo)
matrixs = [matrixone.todense(), matrixtwo.todense()]

#Do Surprisal Analysis

obj1 = SurprisalAnalysis(matrix_type = "sparse", normalize = "counts", var_method="bootstrap")
obj1.calculate_surprisals(*matrices)
obj1.calculate_surprisals_var(*matrices)
obj1.calculate_JSDs(*matrices)
y1 = np.array(obj1.surprisals_var_)
weights1 = np.array(obj1.surprisal_weights_)

#y1, total = getUncertainty(surpvars1, weights1)

obj2 = SurprisalAnalysis(matrix_type = "sparse", normalize = "counts", var_method="analytical")
obj2.calculate_surprisals(*matrices)
obj2.calculate_surprisals_var(*matrices)
obj2.calculate_JSDs(*matrices)
y2 = np.array(obj2.surprisals_var_)
weights2 = np.array(obj2.surprisal_weights_)

#y2, total = getUncertainty(surpvars2, weights2)

########Dense stuff isn't working

#obj3 = SurprisalAnalysis(matrix_type = "dense", normalize = "counts", var_method="analytical")
#obj3.calculate_surprisals(*matrixs)
#obj3.calculate_surprisals_var(*matrixs)
#obj3.calculate_JSDs(*matrixs)
#y3 = np.array(obj3.surprisals_var_)
#
#
#obj4 = SurprisalAnalysis(matrix_type = "dense", normalize = "counts", var_method="bootstrap")
#obj4.calculate_surprisals(*matrixs)
#obj4.calculate_surprisals_var(*matrixs)
#obj4.calculate_JSDs(*matrixs)
#y4 = np.array(obj4.surprisals_var_)


#Make plots
x = range(150)

fig, ax = plt.subplots()

plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')

plt.xlim(10**-6, 10**-1)
plt.ylim(10**-6, 10**-1)

#Plot with bootstrap as x and analytical as y

ax.scatter(y1, y2, color = 'red', label = 'Sparse Analytical') 
#ax.scatter(x, y2, color = 'blue', label = 'Sparse Bootstrap')
#ax.scatter(x, y3, color = 'green', label = 'Dense Analytical')
#ax.scatter(x, y4, color = 'purple', label = 'Dense Bootstrap')

legend = ax.legend()
plt.show()

fig.savefig('Round3ff96/10kcompvars750.png')
