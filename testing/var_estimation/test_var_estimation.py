import os, sys
sys.path.append('../../src')

from surprisal import SurprisalAnalysis
import numpy as np

var_analytic = []
var_bootstrap = []

var_analytic_unnormalized = []
var_bootstrap_unnormalized = []

sa = SurprisalAnalysis(var_method = "analytical")
sb = SurprisalAnalysis(var_method = "bootstrap") 

print '#trial\tncounts\tnstates\tvar_analytic\tvar_bootstrap'
for trial in range(100):

    # randomly choose a number of counts between 1 and 1000000  
    ncounts = int(10.0**(5*np.random.random() + 1))

    # randomly choose a number of multinomial events between 2 and 200
    nstates = 2 + int(10.0**(2.0*np.random.random())*2.0) 

    # make up some fake probabilities of each event for system 1 and 2
    p1 = np.random.random(nstates) 
    p1 = p1/p1.sum()
    p2 = np.random.random(nstates)
    p2 = p2/p2.sum()

    print trial, ncounts, nstates, 

    # draw ncounts from this multinomial 
    c1 = np.random.multinomial(ncounts, p1)
    c2 = np.random.multinomial(ncounts, p2)

    print 'c1', c1, 'c2', c2 
    # compute the variance of the surprisal using bootstrap
    #print 's._compute_si(c1, c2)', s._compute_si([c1, c2])


    var_analytic.append( sa._compute_si_var_analytical([c1, c2], None) )
    var_bootstrap.append( sb._compute_si_var_bootstrap([c1, c2], state_id=None, n_bootstraps=10000)  )

    # var_analytic_unnormalized.append( surprisal_var(c1, c2, bootstrap=False, normalize=False) )
    # var_bootstrap_unnormalized.append( surprisal_var(c1, c2, bootstrap=True, n_bootstraps=1000, normalize=False) )

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
plt.xlim(1e-9, 1e-2)
plt.ylim(1e-9, 1e-2)
plt.xlabel('$\sigma_s^2$ analytical estimate')
plt.ylabel('$\sigma_s^2$ bootstrap estimate')
plt.tight_layout()

#plt.subplot(2,2,2)
#plt.title('var of normalized surprisal $s = \mathcal{S}/N$')
#plt.plot(var_analytic, var_bootstrap,'.')
#plt.plot([1e-9, 1.], [1e-9, 1.,], 'k-')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(1e-9, 1e-2)
#plt.ylim(1e-9, 1e-2)
#plt.xlabel('$\sigma_s^2$ analytical estimate')
#plt.ylabel('$\sigma_s^2$ bootstrap estimate')



#plt.show()
plt.savefig('surprisal_analytical.pdf')


"""
p = np.array([  2.,   5.,  30.,  10.])
q = np.array([  3.,   3.,  3.,  3.])
surprisal_var(p,q, bootstrap=False)
surprisal_var(p,q, bootstrap=True)
q = q*100
p = p*100
surprisal_var(p,q, bootstrap=True)
surprisal_var(p,q, bootstrap=False)
"""


