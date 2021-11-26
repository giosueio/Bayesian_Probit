# Giosuè Migliorini

# MSc in Data Science - Bocconi University

# 20592 Statistics and Probability

# Final project

import numpy as np
import numpy.linalg as la
import scipy.stats as stats
import bayesian_probit as bprobit

np.random.seed(1)


# Simulated dataset

k = 3 #number of variables
n = 200 #number of observations

X = np.random.normal(size=(n, k))
true_beta = np.array([1,1,1])
p_true = stats.multivariate_normal.cdf(X @ true_beta) # parameter p of the Bernoulli dist. of Y
Y = stats.bernoulli.rvs(p_true)

t = 1000 # iterations of each algorithm
nchain = 4 # number of short chains for gibbs_sampler_shortchains

print(f'The true value of the parameters is: {true_beta}')


# Initial values of the parameters are the least square estimates

b = la.inv(X.T @ X) @ X.T @ Y

# Choice of priors between 'flat' and 'conjugate'

prior = 'flat'

# Mean and variance of conjugate Normal prior

muprior = np.array([10,-10,10])
V = np.diag(np.ones(k))

# Run the scripts

metropolis = bprobit.mh_norm(b, X, Y, true_beta, t, prior=prior, muprior=muprior, V=V)
gibbs = bprobit.gibbs_sampler(X, Y, b, true_beta, t, prior=prior, muprior=muprior, V=V)
gibbs_shortchains = bprobit.gibbs_sampler_shortchains(X, Y, b, nchain, true_beta, t, prior=prior, muprior=muprior, V=V)

metropolis, gibbs, gibbs_shortchains