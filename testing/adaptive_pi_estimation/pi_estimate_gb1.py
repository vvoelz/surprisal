import os, sys
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
for i in range(len(i_indices)):
    C[i_indices[i], j_indices[i]] += 0.01

### Reversible estimation of the equilibirum distribution  ###
pi_i_guess = np.ones( nstates )/float(nstates) # an initial guess

pi_i = estimate_pi_i(C, pi_i=pi_i_guess)

from matplotlib import pyplot as plt
plt.figure( figsize=(6, 6) )

for trial in range(10):

    if (1):
      # perform a round of uniform sampling
      nsamples = 10
      for i in range(nstates):
        jump_indices = jumps[i][0]
        jump_probs = jumps[i][1]
        sampled = np.random.multinomial(nsamples, jump_probs).astype(float)
        for j in range(len(sampled)):
            C[i, jump_indices[j]] += sampled[j]

      # ... and estimate pi_i
      pi_i = estimate_pi_i(C, pi_i=pi_i_guess)

      print 'trial', trial, 'samples', trial*nsamples, 'estimated pi_i', pi_i[0:10], '...'
      plt.plot(true_pi, pi_i, '.', label='%d samples/state'%(trial*nsamples))


    else:
      # perform sampling proportional to the true equilibrium pops
      nsamples = np.array([max(10*nstates*true_pi[i], 1) for i in range(nstates)])
      for i in range(nstates):
        jump_indices = jumps[i][0]
        jump_probs = jumps[i][1]
        sampled = np.random.multinomial(nsamples[i], jump_probs).astype(float)
        for j in range(len(sampled)):
            C[i, jump_indices[j]] += sampled[j]

      # ... and estimate pi_i
      pi_i = estimate_pi_i(C, pi_i=pi_i_guess)

      print 'trial', trial, 'samples', trial*nsamples.sum(), 'estimated pi_i', pi_i[0:10], '...'
      plt.plot(true_pi, pi_i, '.', label='%d samples'%(trial*nsamples.sum()))


# re-estimate pi_i using MLE
sampled_T = np.copy(C) 
for i in range(nstates):
    sampled_T[i,:] = C[i,:]/C[i,:].sum()
mle_pi = get_eigenvectors(sampled_T, 1)[1][:, 0]
plt.plot(true_pi, mle_pi, '.', label='MLE from counts')

# estimate pi_i as stationary eigenvector of reversible T from counts
rev_T =  np.copy(sampled_T)
C_i = C.sum(axis=1)
for i in range(nstates):
  for j in range(nstates):
    rev_T[i,j] = (C[i,j]+C[j,i])*pi_i[j]/(C_i[i]*pi_i[j] + C_i[j]*pi_i[i])
mle_rev_pi = get_eigenvectors(rev_T, 1)[1][:, 0]
plt.plot(true_pi, mle_rev_pi, '.', label='MLE from rev_T')

#plt.title('%d samples'%n)
plt.plot([1e-5,1.], [1e-5,1.], 'k-')  # plot the diagonal
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=8, loc='best')
plt.xlabel('true $\\pi_i$')
plt.ylabel('estimated $\\pi_i$')
plt.tight_layout()
#plt.show()
plt.savefig('pi_estimates_GB1.pdf')

