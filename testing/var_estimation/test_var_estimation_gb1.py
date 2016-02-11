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

# get the equilibirium populations, pi_i
from msmbuilder.msm_analysis import get_eigenvectors
pi = get_eigenvectors(T, 1)[1][:, 0]
print 'pi', pi, 'pi.sum()', pi.sum()

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


# Initialize two count matrices, filled with pseudocounts  nprior*pi_i*T_ij
nprior = 1000
C1 = np.zeros( (nstates,nstates), dtype=float )
C2 = np.zeros( (nstates,nstates), dtype=float )
for i in range(len(i_indices)):
    C1[i_indices[i], j_indices[i]] = nprior*pi[i_indices[i]]*values[i]
    C2[i_indices[i], j_indices[i]] = nprior*pi[i_indices[i]]*values[i]





from matplotlib import pyplot as plt
plt.figure( figsize=(8, 8) )

panel = 0

nsamples = [10,100,1000,10000]
for n in nsamples:

    panel += 1
    # Initialize two count matrices, filled with pseudocounts  nprior*pi_i*T_ij
    nprior = 150
    C1 = np.zeros( (nstates,nstates), dtype=float )
    C2 = np.zeros( (nstates,nstates), dtype=float )
    for i in range(len(i_indices)):
        C1[i_indices[i], j_indices[i]] = 0.01
        C2[i_indices[i], j_indices[i]] = 0.01  # nprior*pi[i_indices[i]]*values[i]


    var_analytic = []
    var_bootstrap = []

    var_analytic_nzcounts = []
    var_bootstrap_nzcounts = []

    sa = SurprisalAnalysis(var_method = "analytical")
    sb = SurprisalAnalysis(var_method = "bootstrap")



    print '#trial\tncounts\tnstates\tvar_analytic\tvar_bootstrap'

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

        print 'State', i
        print '\tC1 row', C1[i,0:10], '...'
        print '\tC2 row', C2[i,0:10], '...'

        # this version considers transitions with zero counts, which bootstrap creates a prior for
        var_analytic.append( sa._compute_si_var_analytical([C1[i,:], C2[i,:]], None) )
        var_bootstrap.append( sb._compute_si_var_bootstrap([C1[i,:], C2[i,:]], state_id=None, n_bootstraps=10000)  )

        # this version only sends transitions with nonzero counts
        var_analytic_nzcounts.append( sa._compute_si_var_analytical([sampled1, sampled2], None) )
        var_bootstrap_nzcounts.append( sb._compute_si_var_bootstrap([sampled1, sampled2], state_id=None, n_bootstraps=10000)  )



        print i, n, var_analytic[-1], var_bootstrap[-1]  #, var_analytic_unnormalized[-1], var_bootstrap_unnormalized[-1]

      
    print 'Corrcoef of log(var) =', np.corrcoef(np.log(var_analytic), np.log(var_bootstrap))
    print 'Average std(log10(var))  =', np.mean( (np.log10(var_analytic)-np.log10(var_bootstrap))**2.0 )**0.5


    plt.subplot(2,2,panel)
    plt.title('%d samples'%n)
    plt.plot(var_analytic, var_bootstrap,'.')
    plt.plot(var_analytic_nzcounts, var_bootstrap_nzcounts,'r.')
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
plt.savefig('surprisal_analytical_GB1.pdf')


