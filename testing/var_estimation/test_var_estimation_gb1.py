import os, sys
sys.path.append('../../src')

from surprisal import *
import numpy as np

import scipy
from scipy import io
from scipy.sparse import issparse


# Load in the GB1 hairpin macrostate transition matrix
T = io.mmread('./gb1-tProb.mtx')
nstates = 150

# compile the probabilities of all possible Markov jumps
i_indices, j_indices, values = scipy.sparse.find(T)
jumps = {}  # {i: [jump_indices, jump_probs]
for i in range(nstates):
    jump_indices = j_indices[i_indices==i]
    jump_probs = values[i_indices==i]
    jumps[i] = [jump_indices, jump_probs] 

    print 'State', i
    print '\tjump_indices', jump_indices
    print '\tjump_probs', jump_probs
    print '\tjump_probs.sum()', jump_probs.sum()



from matplotlib import pyplot as plt
plt.figure( figsize=(8, 8) )

panel = 0

nsamples = [10,100,1000,10000]
for n in nsamples:

    panel += 1
    # Initialize two count matrices, filled with pseudocounts 0.01 for connected states
    nprior = 150
    C1 = np.zeros( (nstates,nstates), dtype=float )
    C2 = np.zeros( (nstates,nstates), dtype=float )
    for i in range(len(i_indices)):
        C1[i_indices[i], j_indices[i]] = 0.01
        C2[i_indices[i], j_indices[i]] = 0.01  


    var_analytic = []
    var_bootstrap = []

    var_analytic_nzcounts = []
    var_bootstrap_nzcounts = []

    sa = SurprisalAnalysis([C1,C2])

    print '#trial\tnsamples\tcounts\tvar_analytic\tvar_bootstrap'

    # perform a round of uniform sampling
    for i in range(nstates):
        jump_indices = jumps[i][0]
        jump_probs = jumps[i][1]
        sampled1 = np.random.multinomial(n, jump_probs).astype(float)
        for j in range(len(sampled1)):
            C1[i, jump_indices[j]] += sampled1[j]
        sampled2 = np.random.multinomial(n, jump_probs).astype(float)
        for j in range(len(sampled2)):
            C2[i, jump_indices[j]] += sampled2[j]

        print i, n, 
        print 'C1 =', C1[i,0:5], '...',
        print 'C2 =', C2[i,0:5], '...',

        # this version considers transitions with zero counts, which bootstrap creates a prior for
        var_analytic.append( sa.compute_si_var_analytical([C1[i,:], C2[i,:]]) )
        var_bootstrap.append( sa.compute_si_var_bootstrap([C1[i,:], C2[i,:]], n_bootstraps=1000)  )

        # this version only sends transitions with nonzero counts
        var_analytic_nzcounts.append( sa.compute_si_var_analytical([sampled1, sampled2]) )
        var_bootstrap_nzcounts.append( sa.compute_si_var_bootstrap([sampled1, sampled2], n_bootstraps=1000)  )
        ## convert to arrays
        print var_analytic[-1], var_bootstrap[-1]  #, var_analytic_unnormalized[-1], var_bootstrap_unnormalized[-1]

    ## convert to arrays
    var_analytic, var_bootstrap = np.array(var_analytic), np.array(var_bootstrap)
    var_analytic_nzcounts, var_bootstrap_nzcounts = np.array(var_analytic_nzcounts), np.array(var_bootstrap_nzcounts)
    Ind = (var_analytic > 0.)*(var_bootstrap > 0.)      
    Ind_nz = (var_analytic_nzcounts > 0.)*(var_bootstrap_nzcounts > 0.)

    print 'Corrcoef of log(var) =', np.corrcoef(np.log(var_analytic[Ind]), np.log(var_bootstrap[Ind]))
    print 'Average std(log10(var))  =', np.mean( (np.log10(var_analytic[Ind])-np.log10(var_bootstrap[Ind]))**2.0 )**0.5


    plt.subplot(2,2,panel)
    plt.title('%d samples'%n)
    plt.plot(var_analytic[Ind], var_bootstrap[Ind],'.')
    plt.plot(var_analytic_nzcounts[Ind_nz], var_bootstrap_nzcounts[Ind_nz],'r.')
    plt.legend(['full count row', 'nonzero counts only'], fontsize=8, loc='best')

    plt.plot([1e-9, 1.], [1e-9, 1.,], 'k-')
    plt.xscale('log')
    plt.yscale('log')
    #plt.xlim(1e-7, 1.)
    #plt.ylim(1e-7, 1.)
    plt.xlabel('$\sigma_s^2$ analytical estimate')
    plt.ylabel('$\sigma_s^2$ bootstrap estimate')
    plt.tight_layout()


#plt.show()
outfile = 'surprisal_analytical_GB1.pdf'
plt.savefig(outfile)
print 'Wrote:', outfile
