import os, sys, time
sys.path.append('../../src')

from surprisal import *
import numpy as np

import scipy
from scipy import io
from scipy.sparse import issparse


###### Functions #######

def estimate_pi_i(C, pi_i = None, ftol=1e-12, maxiters=100000):
    """
    Use the self-consistent TRAM estimator of Noe et al to reversibly estimate the equilibrium distribution.

    	pi_i = \sum_j  (C_ij + C_ji)/(C_i/pi_i^(k) + C_j/pi_j)

    Iteration will continue until either the maximum number is reached, or the reduced free energies
    reach a tolerance threshold

    Inputs
    ------
    C          - array of observed counts C_ij from state i to j

    Parameters
    ---------
    pi_i       - an array of initial guesses for the equil distribution
    ftol       - tolerance of the maximum difference in log-populations (free energies)
    maxiters   - mamimum number of self-consistent iterations

    Returns
    -------
    pi_i       - the TRAM estimate of the equilibrium distribution

    """


    nstates = C.shape[0]
    C_i = C.sum(axis=1)

    # If an initial guess is not supplied, use a uniform distribution
    if not ('numpy' in str(type(pi_i))):
        pi_i = np.ones( nstates, dtype='float64')/float(nstates)
    else:
        pi_i = pi_i.astype('float64')

    # helper quantity:  matrix of elements a_ij = C_ij + C_ji
    a = C + C.transpose() 

    pi_i_next = np.copy(pi_i)
    for trial in xrange(maxiters):

        # helper quantity: a matrix of elements b_ij = (\sum_j C_ij)/pi_i + (\sum_i C_ij)/pi_j
        x = C_i/pi_i
        b = np.tile(x.reshape(nstates,1), nstates) + np.tile(x, (nstates,1))

        pi_i_next = (a/b).sum(axis=1)

        f_i, f_i_next = -np.log(pi_i), -np.log(pi_i_next)
        # compute the absolute change of f_i
        df = np.max(np.abs(f_i - f_i_next))
        if trial%100 == 0:
            print '### estimator iter', trial, 'df', df, 'pi_i.sum()', pi_i.sum() #, 'pi_i_next', pi_i_next
        
        pi_i = np.copy(pi_i_next)
        pi_i = pi_i/pi_i.sum()  # make sure populations is normalized
     
        if df < ftol:
            return pi_i

    print 'WARNING: maxiters=%d reached; not converged to ftol = %e'%(maxiters, ftol)
    return pi_i


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

    # update counts and pi_i
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
    print 'Corrcoef of log(var) =', np.corrcoef(np.log(var_analytic[var_analytic>0]), np.log(var_bootstrap[var_bootstrap>0]))
    print 'Average std(log10(var))  =', np.mean( (np.log10(var_analytic)-np.log10(var_bootstrap))**2.0 )**0.5

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

