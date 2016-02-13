import os, sys, time
sys.path.append('../../src')

from surprisal import *
import numpy as np

import scipy
from scipy import io
from scipy.sparse import issparse


###### Functions #######

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


############
# First, perform a round of uniform sampling
nsamples = 10
for i in range(nstates):
    jump_indices = jumps[i][0]
    jump_probs = jumps[i][1]
    sampled = np.random.multinomial(nsamples, jump_probs).astype(float)
    sampled_star = np.random.multinomial(nsamples, jump_probs).astype(float)
    for j in range(len(sampled)):
        C[i, jump_indices[j]] += sampled[j]
        C_star[i, jump_indices[j]] += sampled_star[j]


# update counts (C and C_star) and equil pops (pi_i and pi_i_star)
sa.update([C, C_star])


############
# Next, we  perform surprisal-based sampling 

choose_mode = 'adaptive'
#choose_mode = 'random'
#choose_mode = 'continue'   # continue from the last state transitioned to 

sample_mode = 'uniform'
#sample_mode = 'traj'  

# calculate surprisal variance for all states
surprisal_var = []
for i in range(nstates):
    surprisal_var.append( sa.compute_si_var_analytical([C[i,:], C_star[i,:]]) )
surprisal_var = np.array( surprisal_var )

# estimate the JSD_i variance for all states
pi_i_comb = np.array([sa.pi_i[k] for k in range(len(sa.pi_i))]).mean(axis=0)
JSD_var = surprisal_var*pi_i_comb*pi_i_comb

# sample next the state with the largest JSD uncertainty
if choose_mode == 'adaptive':
    chosen_state  = np.argmax(JSD_var)
else:
    chosen_state = np.random.randint(nstates)

nrounds = 100
total_sample_traj, total_JSD_var_traj = [], []  # keep track of total JSD_var over time
for trial in range(nrounds):

    # sample the chosen state
    nsamples = 20
    if sample_mode == 'uniform':
        jump_indices = jumps[chosen_state][0]
        jump_probs = jumps[chosen_state][1]
        sampled = np.random.multinomial(nsamples, jump_probs).astype(float)
        sampled_star = np.random.multinomial(nsamples, jump_probs).astype(float)
        for j in range(len(sampled)):
            C[chosen_state, jump_indices[j]] += sampled[j]
            C_star[chosen_state, jump_indices[j]] += sampled_star[j]
    else:  # sample_mode == 'traj'
        chosen_state_star = chosen_state
        for step in range(nsamples):
            jump_indices = jumps[chosen_state][0]
            jump_probs = jumps[chosen_state][1]
            jump_indices_star = jumps[chosen_state_star][0]
            jump_probs_star = jumps[chosen_state_star][1]
            sampled = np.random.multinomial(1, jump_probs).astype(float)
            sampled_star = np.random.multinomial(1, jump_probs_star).astype(float)
            j = np.nonzero(sampled)[0][0]
            j_star = np.nonzero(sampled_star)[0][0]
            C[chosen_state, jump_indices[j]] += 1.
            C_star[chosen_state, jump_indices_star[j_star]] += 1.
            chosen_state = jump_indices[j] 
            chosen_state_star = jump_indices_star[j_star] 
            

    # update counts (C and C_star) and equil pops (pi_i and pi_i_star)
    sa.update([C, C_star])

    # recalculate the surprisal variance for *only* the chosen state 
    surprisal_var[chosen_state] = sa.compute_si_var_analytical([C[chosen_state,:], C_star[chosen_state,:]]) 

    # estimate the JSD_i variance for all states
    pi_i_comb = np.array([sa.pi_i[k] for k in range(len(sa.pi_i))]).mean(axis=0)
    JSD_var = surprisal_var*pi_i_comb*pi_i_comb


    # sample next the state with the largest JSD uncertainty
    if choose_mode == 'adaptive':
        chosen_state  = np.argmax(JSD_var)
    elif choose_mode == 'random':
        chosen_state = np.random.randint(nstates)
    else:
        # i.e. choose_mode = 'continue'
        pass 

    total_samples = trial*nsamples
    total_sample_traj.append(total_samples)
    total_JSD_var_traj.append(JSD_var.sum())
    print 'trial', trial, 'samples', total_samples,
    print 'chosen_state', chosen_state, 'JSD_var_i =',  JSD_var[chosen_state],
    print 'total JSD_var', JSD_var.sum()

    if trial%10 == 0:
        plt.subplot(1,2,1)
        plt.plot(range(nstates), JSD_var, '.', label='%d samples'%total_samples)



plt.subplot(1,2,1)
plt.yscale('log')
plt.ylim(1e-12, 1e-5)
plt.legend(fontsize=8, loc='best')
plt.xlabel('state index')
plt.ylabel('JSD_var')

plt.subplot(1,2,2)
plt.plot(total_sample_traj, total_JSD_var_traj, '*-')
#plt.xscale('log')
plt.yscale('log')
plt.xlabel('number of samples')
plt.ylabel('total JSD variance')

plt.tight_layout()
#plt.show()
plt.savefig('%s_%s_GB1.pdf'%(choose_mode,sample_mode))

