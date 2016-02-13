import os, sys, time
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
true_pi = get_eigenvectors(T, 1)[1][:, 0]
print 'true_pi', true_pi, 'true_pi.sum()', true_pi.sum()

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


# Initialize a count matrix, and fill i with ones for connected states
C = np.zeros( (nstates,nstates), dtype=float )
C_star = np.zeros( (nstates,nstates), dtype=float )  # for surprisal analysis
for i in range(len(i_indices)):
    C[i_indices[i], j_indices[i]] += 0.01
    C_star[i_indices[i], j_indices[i]] += 0.01


# Instantiate a SurprisalAnalysis object
# NOTE: This will compute estimates of pi_i and pi_i_star upon instantiation
sa = SurprisalAnalysis([C,C_star])


from matplotlib import pyplot as plt
plt.figure( figsize=(12, 6) )


nsamples = np.array([10,10,30,50]) #,100,300,500])

for trial in range(len(nsamples)):

    # perform a round of uniform sampling
    for i in range(nstates):
        jump_indices = jumps[i][0]
        jump_probs = jumps[i][1]
        sampled = np.random.multinomial(nsamples[trial], jump_probs).astype(float)
        sampled_star = np.random.multinomial(nsamples[trial], jump_probs).astype(float)
        for j in range(len(sampled)):
            C[i, jump_indices[j]] += sampled[j]
            C_star[i, jump_indices[j]] += sampled_star[j]

    # update counts (C and C_star) and equil pops (pi_i and pi_i_star)
    sa.update([C, C_star]) 

    total_samples = nsamples[0:trial+1].sum()
    print 'trial', trial, 'samples', total_samples , 'estimated pi_i', sa.pi_i[0][0:10], '...'
    plt.subplot(1,2,1)
    plt.plot(true_pi, sa.pi_i[0], '.', label='%d samples/state'%(total_samples))


    ### surprisal analysis ###
    var_analytic, var_bootstrap = [], []

    print '#trial\tnsamples\tvar_analytic\tvar_bootstrap\ttime_analytic (s)\ttimer_bootstrap (s)'
    for i in range(nstates):

        start_time = time.clock()
        var_analytic.append( sa.compute_si_var_analytical([C[i,:], C_star[i,:]]) )
        analytic_time = time.clock() - start_time

        start_time = time.clock()
        var_bootstrap.append( sa.compute_si_var_bootstrap([C[i,:], C_star[i,:]], n_bootstraps=1000) )
        bootstrap_time = time.clock() - start_time

        print i, total_samples, var_analytic[-1], var_bootstrap[-1], analytic_time, bootstrap_time

    var_analytic, var_bootstrap = np.array(var_analytic), np.array(var_bootstrap)
    Ind = (var_analytic > 0.)*(var_bootstrap > 0.) # avoid log(0) in plot 
    print 'Corrcoef of log(var) =', np.corrcoef(np.log(var_analytic[Ind]), np.log(var_bootstrap[Ind]))
    print 'Average std(log10(var))  =', np.mean( (np.log10(var_analytic[Ind])-np.log10(var_bootstrap[Ind]))**2.0 )**0.5

    # plot analytical vs bootstrap estimates of surpirsal uncertinaty
    plt.subplot(1,2,2)
    plt.plot(var_analytic, var_bootstrap,'.', label='%d samples/state'%(total_samples))
    plt.plot([1e-9, 1.], [1e-9, 1.,], 'k-')


plt.subplot(1,2,1)
plt.plot([1e-5,1.], [1e-5,1.], 'k-')  # plot the diagonal
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=8, loc='best')
plt.xlabel('true $\\pi_i$')
plt.ylabel('estimated $\\pi_i$')

plt.subplot(1,2,2)
plt.legend(fontsize=8, loc='best')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\sigma_s^2$ analytical estimate')
plt.ylabel('$\sigma_s^2$ bootstrap estimate')

plt.tight_layout()
#plt.show()
plt.savefig('pi_estimates_GB1.pdf')

