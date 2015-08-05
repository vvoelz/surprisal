import os,sys
from scipy.io import mmread
sys.path.append("../src/")
from surprisal import SurprisalAnalysis
import numpy as np
import matplotlib.pyplot as plt

matrices = []
for pid in range(6383,6391):
    matrix = mmread("Fs-%d-tCounts-macro40.mtx"%pid).tolil()
    matrices.append(matrix)

obj = SurprisalAnalysis(matrix_type = "sparse",normalize = "counts",var_method="analytical")
obj.calculate_surprisals(*matrices)
obj.calculate_surprisals_var(*matrices)
obj.calculate_JSDs(*matrices)
print obj.surprisals_
print obj.surprisals_var_
print obj.surprisal_weights_
print obj.JensenShannonDivergence_
print np.argsort(obj.JensenShannonDivergence_)

obj.surprisal_weights_ = np.array(obj.surprisal_weights_)
obj.surprisals_ = np.array(obj.surprisals_)
obj.surprisals_var_ = np.array(obj.surprisals_var_)
if (1):
    plt.figure()
    # plot contours
    sc = np.array([10**i for i in np.arange(-8,0,0.01)])
    D_values = [10**i for i in np.arange(-8,0,1)]
    for D in D_values:
        plt.plot(D/sc, sc, '-', linewidth=1)
    plt.xlim(1e-5, 1.0)
    plt.xscale('log')
    plt.ylim(1e-3, 1.0)
    plt.yscale('log')

    # plot pi_i, s_i data
    Ind = ~np.isnan(obj.surprisals_)
    plt.errorbar(obj.surprisal_weights_[Ind],obj.surprisals_[Ind],obj.surprisals_var_[Ind]**0.5,
                 fmt='ko', markerfacecolor='None', elinewidth=0.5, capsize=1)

    UseLabels = True
    if (UseLabels):
      for i in range(len(obj.surprisals_)):
        if obj.surprisal_weights_[i]*obj.surprisals_[i] > 5e-4:
          print 'State', i, 'pi_i =', obj.surprisal_weights_[i], 's_i =', obj.surprisals_[i], 'pi_i*si_i =', obj.JensenShannonDivergence_[i]
          plt.text(obj.surprisal_weights_[i], obj.surprisals_[i], str(i), fontsize=6, color='r', weight='bold')
    plt.xlabel('$\\bar{\pi}_i$')
    plt.ylabel('$s_i$')
    plt.savefig("Fs_FAH_surprials_all_MSMs.pdf")
    #plt.show()
