import os, sys
sys.path.append('../../src')

from surprisal import SurprisalAnalysis
import numpy as np

var_analytic = []
var_bootstrap = []

# Garett's count matrices after 750 rounds:
mat1 = np.load('mat1.npy')
mat2 = np.load('mat2.npy')

sa = SurprisalAnalysis([mat1,mat2])

nstates = mat1.shape[0]

print '#trial\tncounts\tnstates\tvar_analytic\tvar_bootstrap'
for trial in range(0,nstates):

    c1 = mat1[trial,:]
    for i in range(len(c1)):
        if c1[i] > 0.0:
            c1[i] -= 149./150.0    # fix matrices to have pseudocounts of 1/nstates instead of 1.0  
    c2 = mat2[trial,:]
    for i in range(len(c2)):
        if c2[i] > 0.0:
           c2[i] -= 149./150.0   # fix matrices to have pseudocounts of 1/nstates instead of 1.0 

    np.set_printoptions(precision=3)
    print 'row index', trial, 'c1 =', c1[0:5], '... c2 =', c2[0:5], '...', 

    var_analytic.append( sa.compute_si_var_analytical([c1, c2]) )
    var_bootstrap.append( sa.compute_si_var_bootstrap([c1, c2], n_bootstraps=10000)  )

    print var_analytic[-1], var_bootstrap[-1]  #, var_analytic_unnormalized[-1], var_bootstrap_unnormalized[-1]

   
print 'Corrcoef of log(var) =', np.corrcoef(np.log(var_analytic), np.log(var_bootstrap))
print 'Average std(log10(var))  =', np.mean( (np.log10(var_analytic)-np.log10(var_bootstrap))**2.0 )**0.5

from matplotlib import pyplot as plt

plt.figure( figsize=(5, 4.5) )

#plt.title('var of normalized surprisal $s = \mathcal{S}/N$')
plt.plot(var_analytic, var_bootstrap,'.')
plt.plot([1e-9, 1.], [1e-9, 1.,], 'k-')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-7, 1.)
plt.ylim(1e-7, 1.)
plt.xlabel('$\sigma_s^2$ analytical estimate')
plt.ylabel('$\sigma_s^2$ bootstrap estimate')
plt.tight_layout()


#plt.show()
plt.savefig('surprisal_analytical_mat750.pdf')


