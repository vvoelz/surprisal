#/usr/bin/env python

import sys
import os 
import numpy as np
from scipy.sparse import issparse
from msmbuilder import version

msmb_version = version.version

if msmb_version == '2.8.2':
    from msmbuilder.MSMLib import estimate_transition_matrix
    from msmbuilder.msm_analysis import get_eigenvectors
elif msmb_version == '3.2.0':
    from msmbuilder.msm import MarkovStateModel



#TODO: Add Kullback-Leibler Divergence

def check_matrices_shape(matrices):

    s = matrices[0].shape
    for i in range(len(matrices)):
        if matrices[i].shape != s:
            raise TypeError("Matrix %d has a different shape!"%i)
    return s

def estimate_mle_populations(matrix):
    if msmb_version == '2.8.2':
        t_matrix = estimate_transition_matrix(matrix)
        populations = get_eigenvectors(t_matrix, 1, **kwargs)[1][:, 0]
        return populations
    elif msmb_version == '3.2.0':
        obj = MarkovStateModel()
        populations = obj._fit_mle(matrix)[1]
        return populations



class SurprisalAnalysis:

    def __init__(self, matrix_type = "sparse", normalize = 'counts', var_method = "analytical"):
        """
        Calculates surprisal for arbitrary sets of counts. If normalized, the
        surprisal value is divided by the total number of counts from that state.


        OPTIONS
        normalize: "counts": Equation 7 in Ref[1].
                   "mle": Equation 14 in Ref[1].
                   "None": Equation 6? in Ref[1].

        var_method: "analytical"
                    "bootstrap"

        Ref[1]. Voelz, V. A., Elman, B., Razavi, A. M., & Zhou, G. (2014).
                Surprisal Metrics for Quantifying Perturbed Conformational
                Dynamics in Markov State Models.
        """

        self.matrix_type = matrix_type
        self.normalize = normalize
        self.var_method = var_method

        self.surprisals_ = None
        self.surprisals_var_ = None
        self.surprisal_weights_ = None
        self.state_populations_mle_ = None
        self.state_populations_counts_ = None
        self.JensenShannonDivergence_ = None

    def _prepare_c_row(self, c_row):
        """

        :param matrices:
        :param state_id:

        #return combined counts(1-D array len = number of states),
                total counts per model(1D list, len = number of models)
                total counts combined(float)
        return c_comb(combined counts), total_counts_model, total_comb
        """
        if len(c_row) < 2:
            raise ValueError(' must be given at least two count arrays!')

        # compute combined counts
        c_comb = c_row[0] + c_row[1]
        if len(c_row) > 2:
            for i in range(2, len(c_row)):
                c_comb += c_row[i]
        # compute count totals
        total_counts_model = [c.sum() for c in c_row]
        total_comb = c_comb.sum()

        return c_comb, total_counts_model, total_comb

    def _compute_state_populations_counts(self,matrices):

        self.state_populations_counts_ = []
        for matrix in matrices:
            self.state_populations_counts_.append(np.array(matrix.sum(axis=1)).flatten())
        self.state_populations_counts_ = np.array(self.state_populations_counts_)

    def _compute_state_populations_mle(self,matrices):

        self.state_populations_mle_ = []
        for matrix in matrices:
            if issparse(matrix):
                self.state_populations_mle_.append(estimate_mle_populations(matrix.toarray()))
            else:
                self.state_populations_mle_.append(estimate_mle_populations(matrix))
        self.state_populations_mle_ = np.array(self.state_populations_mle_)

    def _compute_s_weights(self, matrices, normalize="counts"):

        if normalize == "counts":
            #Count-based estimation of combined populations used in JSD. Eq 10 in Ref[1].
            if self.state_populations_counts_ is None:
                self._compute_state_populations_counts(matrices)
            return np.sum(self.state_populations_counts_,axis=0)/np.sum(self.state_populations_counts_)

        elif normalize == "mle":
            #Eigenvector-based estimation of combined populations used in JSD. Eq 14,15 in Ref[1].
            if self.state_populations_mle_ is None:
                self._compute_state_populations_mle(matrices)
            return np.mean(self.state_populations_mle_,axis=0)

    def _compute_count_normalized_si(self, c_row):
        c_comb, totals, total_comb = self._prepare_c_row(c_row)

        si = H(c_comb)
        for i in range(len(c_row)):
            si -= totals[i]/total_comb*H(c_row[i])

        return si

    def _compute_mle_normalized_si(self, c_row, state_id):

        c_comb, totals, total_comb = self._prepare_c_row(c_row)

        si = H(c_comb)
        for i in range(len(c_row)):
            si -= self.state_populations_mle_[i,state_id]/self.state_populations_mle_.sum(axis=0)[state_id]*H(c_row[i])

        return si

    def _compute_unnormalized_si(self, c_row):

        c_comb, totals, total_comb = self._prepare_c_row(c_row)
        si = total_comb*H(c_comb)
        for i in range(len(c_row)):
            si -= totals[i]*H(c_row[i])

        return si

    def _compute_si(self, c_row, state_id=None, normalize="counts"):

        """
        Calculates surprisal for arbitrary sets of counts. If normalized, the
        surprisal value is divided by the total number of counts from that state.
        INPUTS
        c1, c2, ....   any number of numpy counts

        """

        if normalize == "counts":
            return self._compute_count_normalized_si(c_row=c_row)

        elif normalize == "mle":
            if state_id is None:
                raise ValueError("Have to provide state id in order to use MLE population weights")
            else:
                return self._compute_mle_normalized_si(c_row=c_row,state_id=state_id)

        elif (normalize.lower() == "none")|(normalize == None):
            return self._compute_unnormalized_si(c_row=c_row)

    def _compute_si_var_bootstrap(self, c_row, state_id, n_bootstraps=100, normalize="counts"):

        c_comb, totals, total_comb = self._prepare_c_row(c_row)

        # use MLE probs of multinomal distribution (pseudocount of 1/2 according to Jeffrey's prior)
        if normalize == "counts":
            p_list = [counts2probs(c+0.5) for c in c_row]
        elif normalize == "mle":
            p_list = [counts2probs(c+0.5) for c in c_row]

        # Draw bootstrapped counts from a multinomial distribution
        si = np.zeros(n_bootstraps)
        resampled_all = []
        for i in range(len(c_row)):
            resampled_all.append(np.random.multinomial(totals[i], p_list[i], size=n_bootstraps))
        for trial in range(n_bootstraps):
            si[trial] = self._compute_si([resampled[trial,:] for resampled in resampled_all],
                                         state_id,
                                         normalize)

        return si.var()

    def _compute_si_var_analytical(self, c_row, state_id, normalize="counts"):

        c_comb, totals, total_comb = self._prepare_c_row(c_row)

        V = []
        m = self.matrix_shape[0]
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
                q[ndx] = -1.0*self._compute_si(c_row,state_id,normalize)
                if np.sum([c[j] for c in c_row]) >0:
                    q[ndx] -= np.log(float(np.sum([c[j] for c in c_row]))/total_comb)
                if c_row[i][j] > 0:
                    q[ndx] += np.log(float(c_row[i][j])/totals[i])
        if normalize == "counts":
            q = q/total_comb

        return np.dot(q, W.dot(q))

    def cal_si_dense(self, matrices, state_id, normalize="counts"):

        return self._compute_si([matrices[i][state_id] for i in range(len(matrices))], state_id,normalize)

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


def counts2probs(c):
    # convert to float if necessary
    if c.dtype.kind == 'i':
        c = c.astype('float')
    return c/c.sum()

def H(p, normalize="counts"):
    """Returns the entropy H = \sum_i - p_i ln p_i of a distribution.
    INPUTS
    p	        numpy array of values (discrete counts work too)
    OPTIONS
    normalize 	If True, will normalize the input values so that \sum_i p_i = 1
    """

    if normalize == "counts":
        p = counts2probs(p)

    # non-zero entries only
    Ind = (p>0)

    return np.dot(-p[Ind], np.log(p[Ind]))

def H_cross(p,q,normalize=True):

    if normalize:
        p = counts2probs(p)
        q = counts2probs(q)

    # non-zero entries only
    Ind = (p>0)
    return np.dot(-p[Ind], np.log(q[Ind]))

def H_var(c):
    """Estimate the variance of the entropy H due to finite sampling.
    INPUT
    c     	a row of transition counts
    OUTPUT
    var_H	an estimate of the variance of H(c)
    """

    # build MVN/multinomial covariance matrices for c1 and c2
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
    """Returns the covariance matrix for a multinomal sample of counts c."""
    m = c.shape[0]
    total_c = c.sum()
    c_hat = c+0.5
    p_hat = c_hat/c_hat.sum()
    V = np.zeros( (m,m) )
    for j in range(m):
        for k in range(m):
            if j==k:
                V[j,k] = total_c*p_hat[j]*(1.0-p_hat[j])
            else:
                V[j,k] = -total_c*p_hat[j]*p_hat[k]
    return V

