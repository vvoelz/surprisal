#!/usr/bin/env python

# Author: Vincent Voelz (voelz@temple.edu)
# Co-author: Guangfeng Zhou 
# Copyright (c) 2016, Temple University and the Authors
# All rights reserved.


import sys
import os 
import numpy as np
from scipy.sparse import issparse

"""
### NOTES on VAV edits 2/2016 ###

* msmbuilder routines have been stripped out in favor of the Noe DTRAM estimators
  for equilibrium distribution pi_i and transition matrix elements  

* all count arrays are floats by default, allowing fractional counts (e.g. pseudocounts)

* all pseudocount values (for multinomial bootstrapping, covariance estimates, etc.) are now set
  to 1/nstates as the default 

* The SurprisalAnalysis object is initialized with a list of count matrices, and equilibrium distributions
  pi_i are estimated upon instantiation.   Routines are provided to update as necessary. 

* some SurprisalAnalysis attributes have been removed:
    matrix_type <-- these can be inspected on the fly
    var_method <-- analytical/bootstrap are now set as default parameters, which the user can change if desired

* gone are "private" functions (starting with an underscore).  We want access to all the internal functionality

* got rid of counts2probs -- counts are stored as floats by default now, and c/c.sum() does not require its
  own routine

TODOs:

* deal with sparse count matrices!

* fix the following routines :
    def calculate_surprisals(self):
    def calculate_surprisals_var(self):
    def calculate_JSDs(selfs):


"""



### Functions ###


def H(p, normalized=True):
    """Returns the entropy H = \sum_i - p_i ln p_i of a distribution.

    Input
    -----
    p	        numpy array of values (discrete counts work too)

    Parameters
    ----------
    normalized 	If True, will normalize the input values so that \sum_i p_i = 1. Default: True
    """

    if normalized:
        p = p/float(p.sum())   # casts to float

    # use non-zero entries only to avoid log(0)
    Ind = (p>0)

    return np.dot(-p[Ind], np.log(p[Ind]))


def H_cross(p, q, normalized=True):
    """Returns the cross-entropy H(p,q) = \sum_i - p_i ln q_i of a distribution."""

    if normalized:
        p = p/float(p.sum)
        q = q/float(q.sum)

    # non-zero entries only to avoid log(0)
    Ind = (p>0)
    return np.dot(-p[Ind], np.log(q[Ind]))


def H_var(c):
    """Estimate the variance of the entropy H due to finite sampling.

    Input
    ----
    c     	a row of transition counts as a 1D array

    Output
    ------
    var_H	an estimate of the sample variance of H(c)

    """

    # build MVN/multinomial covariance matrix for c
    V = cov_multinomial_counts(c)

    # compute the vector of sensitivities q = [dH/dnj ... ]
    n = c.sum()
    print 'c =', c, 'n =', n
    q = H(c)*np.ones( c.shape )
    for i in range(c.shape[0]):
        if c[i] > 0:
            q[i] -= np.log( float(c[i])/n )
    q = q/n

    # return q^T V q
    return np.dot(q, V.dot(q))


def cov_multinomial_counts(c):
    """Returns the covariance matrix for a multinomal sample of counts c.

    Input
    ----
    c           a row of transition counts as a 1D array

    Output
    ------
    cov_C       the estimated covariance matrix
    """

    m = c.shape[0]
    total_c = c.sum()

    prior_counts = 1.0/float(m)   # VAV 2/2016
    # prior_counts = 0.5          # as in Voelz et al 2014
    c_hat = c + prior_counts 
    p_hat = c_hat/c_hat.sum()
   
    if (0):  # old way, uses for loops
      V = np.zeros( (m,m) )
      for j in range(m):
        for k in range(m):
            if j==k:
                V[j,k] = total_c*p_hat[j]*(1.0-p_hat[j])
            else:
                V[j,k] = -total_c*p_hat[j]*p_hat[k]

    else:  # VAV: numpy routines are faster than for loops
        W = -total_c*np.tile(p_hat.reshape(m,1), m)
        X = np.tile(p_hat, (m,1)) 
        V = W*X 
        V = V - np.diag(np.diag(V)) + np.diag(total_c*p_hat*(1.0-p_hat))
      
    return V


def check_matrices_shape(matrices):

    s = matrices[0].shape
    for i in range(len(matrices)):
        if matrices[i].shape != s:
            raise TypeError("Matrix %d has a different shape!"%i)
    return s


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

        # compute the absolute change in free energies f_i
        f_i, f_i_next = -np.log(pi_i), -np.log(pi_i_next)
        df = np.max(np.abs(f_i - f_i_next))

        # DEBUG
        # if trial%100 == 0:
        #     print '### estimator iter', trial, 'df', df, 'pi_i.sum()', pi_i.sum() #, 'pi_i_next', pi_i_next
        
        pi_i = np.copy(pi_i_next)
        pi_i = pi_i/pi_i.sum()  # make sure populations is normalized
     
        if df < ftol:
            return pi_i

    print 'WARNING: maxiters=%d reached; not converged to ftol = %e'%(maxiters, ftol)
    return pi_i


def estimate_pi_i_sparse(C, pi_i = None, ftol=1e-12, maxiters=100000):
    """
    Use the self-consistent TRAM estimator of Noe et al to reversibly estimate the equilibrium distribution.
    *** designed for sparse arrays ***
    """

    return




#### Classes #####

class SurprisalAnalysis:

    def __init__(self, counts, estimate_pi=True):
        """
        An object for calculating surprisal quantities for arbitrary sets of counts. 

        Inputs
        ------
        counts      - a list of 2D count arrays [C1, C2, ...] where each contain elements C_ij as the
                      number of transitions counted from state i to j

        Parameters
        ---------
        estimate_pi - If True, equilibrium populations pi_i will be estimated for each
                       set of counts upon instantiation,  Default: True

        Ref[1]. Voelz, V. A., Elman, B., Razavi, A. M., & Zhou, G. (2014).
                Surprisal Metrics for Quantifying Perturbed Conformational
                Dynamics in Markov State Models.
        """

        self.nmodels = len(counts)
        self.nstates = counts[0].shape[0]

        # Check that all count matrices have the same dimensions 
        if not self.same_dims(counts):
            raise TypeError("Count matrices do not have the same shape!")

        self.counts = self.nmodels*[None]
        self.pi_i = self.nmodels*[None]
        self.matrix_type = self.nmodels*[None]

        # store the counts and matrix type ('sparse', 'dense') of each
        self.update_counts(counts)
        
        # estimate pi_i for each model   
        if estimate_pi:
            self.update_pi_i()

        # intialize values to be computed later
        self.surprisals = None          # a 1D array of length nstates
        self.surprisals_var = None      # a 1D array of length nstates
        self.surprisal_weights = None   # a 1D array of length nstates
        self.JensenShannonDivergence = None



    def same_dims(self, counts):
        """Check that all count matrices have the same dimensions."""

        s = counts[0].shape
        for i in range(len(counts)):
            if counts[i].shape != s:
                return False
        return True


    def prepare_c_row(self, c_row):
        """
        Prepares the combined and total counts 

        Inputs
        ------
        c_row  - a list (or iterable container) of count rows (1D arrays) for each model

        Returns
        -------
        c_comb             - combined counts (1-D array of length nstates)
        total_counts_model - total counts per model (1D array of length nmodels)
        total_comb         - total counts combined (float)
        """

        if len(c_row) < 2:
            raise ValueError(' must be given at least two count arrays!')

        # compute combined counts
        c_comb = c_row[0] + c_row[1]
        if len(c_row) > 2:
            for i in range(2, len(c_row)):
                c_comb += c_row[i]
        # compute count totals
        total_counts_model = np.array([c.sum() for c in c_row])
        total_comb = c_comb.sum()

        return c_comb, total_counts_model, total_comb


    def update(self, counts):
        """Updates the stored counts and pi_i estimates for each.

        Inputs
        ------
        counts      - a list of 2D count arrays [C1, C2, ... C_K] where each contain elements C_ij as the
                      number of transitions counted from state i to j
        """

        self.update_counts(counts)
        self.update_pi_i()


    def update_counts(self, counts):
        """Updates the stored counts."""

        # determine whether each 2D count array is dense or sparse
        self.matrix_type = []  # 
        for C in counts:
            if issparse(C):
                self.matrix_type.append('sparse')
            else:
                self.matrix_type.append('dense')

        # make sure each 2D count array are floats
        for k in range(len(counts)):
            self.counts[k] = counts[k].astype(float)


    def update_pi_i(self):
        """Update the estimates of equilibirium populations pi_i."""

        # the counts need to be already defined for this to work
        assert len(self.counts) > 0
        assert ('numpy' in str(type(self.counts[0])))

        for k in range(self.nmodels):
            self.pi_i[k] = estimate_pi_i(self.counts[k]) 


    def compute_s_weights(self, method="counts"):
        """Return weights for calculating the total surprisal.

        The "total surprisal" is defined in Eq. 10 of ref[1]:

            s = w_i s_i

        where the weights w_i are based on the observed counts (method="counts"),
        N_i = \sum_j C_{ij}, and N = \sum_ij N_i
   
            w_i = \sum_i [(\sum_k N_i^(k) )/(\sum_k N^(k) )]

        There may also be times when pi_i-based weights are preferred (method="pi_i")

            w_i = \sum_i [(\sum_k pi_i^(k) )/K
        """

        if method == "counts":
            N_k_i = np.array([C.sum(axis=1) for C in self.counts])  
            N_i = N_k_i.sum(axis=0)
            return N_i/N_i.sum()

        elif method == "pi_i":
            pi_k_i = np.array(self.pi_i)
            pi_i = pi_k_i.sum(axis=0)
            return pi_i/pi_i.sum()

        

    def compute_si(self, c_row, normalized=True, method="counts"):
        """Returns the (normalized) surprisal (Eq. 7, Ref[1]) for a set of a row counts.

        Input
        -----
        c_row  - a list (or iterable container) of count rows (1D arrays) for each model

        Paramters
        ---------
        normalized  - If True, return the normalized surprisal.  This is what we want in most cases,
                      and what the "surprisal" typically refers to.   Default: True

        methods     - "counts": weighs entropy contributions by the numbers of observed counts.  This
                      is usually the desired quantity.  Default: "counts".

                    - "pi_i": weights entropy contributions by equilibrium populations.  This can
                      be useful for estimating JS divergences, for example.  For this to work, 
                      the populations pi_i for each model k *must* be already stored.
        """

        c_comb, totals, total_comb = self.prepare_c_row(c_row)
        if normalized:
            si = H(c_comb)
        else:
            si = total_comb*H(c_comb)

        if method == "counts":
            for k in range(len(c_row)):
                if normalized:
                    si -= totals[k]/total_comb*H(c_row[k])
                else:
                    si -= totals[k]*H(c_row[k])

        elif method == "pi_i":
            if self.pi_i[0] == None:
                raise Exception, 'numpy arrays self.pi_i must be stored when using method "pi_i" '
            else:
                for k in range(len(c_row)):
                    if normalized:
                        total_pi_i = np.array([self.pi_i[k][i] for k in range(self.nmodels)]).sum()
                        si -= self.pi_i[k][i]/total_pi_i*H(c_row[k])
                    else:
                        si -= self.pi_i[k][i]*H(c_row[k])
        else:
            raise Exception, 'Unrecognized method.'

        return si


    def compute_si_var_bootstrap(self, c_row, n_bootstraps=1000):
        """Estimate the variance of surprisal si_i using a bootstrap resampling method.

        Input
        -----
        c_row  - a list (or iterable container) of count rows (1D arrays) for each model

        Paramters
        ---------
        n_bootstraps - the number of bootstrap resampling iterations to perform.  Default: 1000
        """

        c_comb, totals, total_comb = self.prepare_c_row(c_row)

        # VAV added: use 1/nstates as a prior
        nstates = float(c_row[0].shape[0])
        prior_counts = 1.0/nstates

        # model the parent multinomial distribution as observed frequencies + pseudocounts
        p_list = [(c+prior_counts)/float((c+prior_counts).sum()) for c in c_row]

        # Draw bootstrapped counts from a multinomial distribution
        si = np.zeros(n_bootstraps)
        resampled_all = []
        for i in range(len(c_row)):
            resampled_all.append(np.random.multinomial(totals[i], p_list[i], size=n_bootstraps).astype(float))
        for trial in range(n_bootstraps):
            si[trial] = self.compute_si([resampled[trial,:] for resampled in resampled_all])

        return si.var()


    def compute_si_var_analytical(self, c_row, normalized=True):

        c_comb, totals, total_comb = self.prepare_c_row(c_row)

        V = []
        #m = self.matrix_shape[0]
        m = len(c_row[0])
        for i in range(len(c_row)):
            V.append(cov_multinomial_counts(c_row[i]))

        # make a block diagonal matrix W = diag(V1, V2)
        W = np.zeros((len(c_row)*m, len(c_row)*m))
        for i in range(len(c_row)):
            W[i*m:i*m+m,i*m:i*m+m] = V[i]

        # compute the vector of sensitivities q = [ds/dnj ... | ds/dn*j ...]
        q = np.zeros(len(c_row)*m)
        for i in range(len(c_row)):
            for j in range(m):
                ndx = j+i*m
                q[ndx] = -1.0*self.compute_si(c_row)
                if np.sum([c[j] for c in c_row]) >0:
                    q[ndx] -= np.log(float(np.sum([c[j] for c in c_row]))/total_comb)
                if c_row[i][j] > 0:
                    q[ndx] += np.log(float(c_row[i][j])/totals[i])

        if normalized:
            q = q/total_comb

        return np.dot(q, W.dot(q))


    """
    def cal_si_dense(self, matrices, state_id, normalize="counts"):

        return self.compute_si([matrices[i][state_id] for i in range(len(matrices))], state_id,normalize)

    def cal_si_sparse(self, matrices, state_id,normalize="counts"):

        return self._compute_si([matrices[i][state_id].toarray()[0] for i in range(len(matrices))], state_id, normalize)

    def cal_si_var_bootstrap_dense(self, matrices, state_id, n_bootstraps=100,normalize="counts"):

        return self._compute_si_var_bootstrap([matrices[i][state_id] for i in range(len(matrices))],
                                              state_id, n_bootstraps, normalize)

    def cal_si_var_bootstrap_sparse(self, matrices, state_id, n_bootstraps=100,normalize="counts"):

        return self._compute_si_var_bootstrap([matrices[i][state_id].toarray()[0] for i in range(len(matrices))],
                                              state_id, n_bootstraps, normalize)

    def cal_si_var_analytical_dense(self, matrices, state_id, normalize="counts"):

        return self._compute_si_var_analytical([matrices[i][state_id] for i in range(len(matrices))],
                                              state_id, normalize)

    def cal_si_var_analytical_sparse(self, matrices, state_id, normalize="counts"):

        return self._compute_si_var_analytical([matrices[i][state_id].toarray()[0] for i in range(len(matrices))],
                                              state_id, normalize)

    """

    def estimate_s_weights(self, matrices, normalize="counts"):
        #Should work for both sparse and dense matrix type
        self.surprisal_weights_ = self._compute_s_weights(matrices, normalize)


    def calculate_surprisals(self, *matrices):

        self.surprisals_ = []
        self.matrix_shape = check_matrices_shape(matrices)

        if self.matrix_type.lower() == "dense":
            calStateSurprisal = self.cal_si_dense
        else:
            calStateSurprisal = self.cal_si_sparse

        if self.normalize.lower() == "mle":
            if self.state_populations_mle_ is None:
                self._compute_state_populations_mle(matrices)

        for state_id in range(self.matrix_shape[0]):
            self.surprisals_.append(calStateSurprisal(matrices=matrices,
                                                      state_id=state_id,
                                                      normalize=self.normalize.lower()))

    def calculate_surprisals_var(self, *matrices):

        self.surprisals_var_ = []
        self.matrix_shape = check_matrices_shape(matrices)

        if self.normalize.lower() == "mle":
            if self.state_populations_mle_ is None:
                self._compute_state_populations_mle(matrices)

        if self.var_method.lower() == "bootstrap":
            if self.normalize.lower() != "counts":
                print "Only count based normalization is available currently."
                print "Use counts based normalization instead."

            if self.matrix_type.lower() == "dense":
                calStateSurprisalVariance = self.cal_si_var_bootstrap_dense
            else:
                calStateSurprisalVariance = self.cal_si_var_bootstrap_sparse

            for state_id in range(self.matrix_shape[0]):
                self.surprisals_var_.append(calStateSurprisalVariance(matrices=matrices,
                                                                      state_id=state_id,
                                                                      n_bootstraps=100,
                                                                      normalize="counts"))
        elif self.var_method.lower() == "analytical":
            if self.normalize.lower() != "counts":
                print "Only count based normalization is available currently."
                print "Use counts based normalization instead."

            if self.matrix_type.lower() == "dense":
                calStateSurprisalVariance = self.cal_si_var_analytical_dense
            else:
                calStateSurprisalVariance = self.cal_si_var_analytical_sparse
            for state_id in range(self.matrix_shape[0]):
                self.surprisals_var_.append(calStateSurprisalVariance(matrices=matrices,
                                                                      state_id=state_id,
                                                                      normalize="counts"))

    def calculate_JSDs(self, *matrices):

        if self.surprisals_ is None:
            self.calculate_surprisals(*matrices)

        if self.surprisal_weights_ is None:
            self.estimate_s_weights(matrices,self.normalize)

        self.JensenShannonDivergence_ = []
        for i in range(self.matrix_shape[0]):
            JSD = self.surprisals_[i]*self.surprisal_weights_[i]
            self.JensenShannonDivergence_.append(JSD)


