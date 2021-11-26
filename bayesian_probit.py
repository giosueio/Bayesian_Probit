import numpy as np
import numpy.linalg as la
import scipy.stats as stats
import matplotlib.pyplot as plt

# Metropolis Hastings
def mh_norm(b0, x, y, true_beta, t = 1000, mvar = 10, *, prior = 'flat', muprior = 0, V = 0):
    
    def logposterior(b): # The proposal distribution - We chose the log scale for easier computation
        lh = 1
        for i in range(np.shape(y)[0]):
            lh *= stats.bernoulli.pmf(y[i], stats.norm.cdf(x[i] @ b)) 
            loglh = np.log(lh)
        if prior == 'conjugate':
            logposterior = loglh + np.log(stats.multivariate_normal.pdf(b, muprior, V)) # in case of conjugate prior we add the pdf of the prior distribution evaluated at the current update
        elif prior == 'flat':
            logposterior = loglh # in case of flat prior we only consider the log likelihood
        return logposterior


    def f_info(b):
        p = stats.multivariate_normal.cdf(x @ b)        
        W = np.diag(1 / (p*(1 - p)) * (1/np.sqrt(2*np.pi)) * np.exp((-(x @ b)**2)/2)) # W = 1/var(Y) * (pdf = derivative of mu wrt eta)^2 ; Variance of Y is evaluated at the current update of the parameters p
        fisher_info = x.T @ W @ x
        return fisher_info

    sample = []
    a = 0
    tstar = 100
    for i in range(t):
        q = np.random.multivariate_normal(b0, mvar * la.inv(f_info(b0))) # sample from proposal
        alfa = min([1, np.exp(logposterior(q)) / np.exp(logposterior(b0))]) # acceptance probability
        u = np.random.random()
        if u < alfa:
            b = q
            a += 1
        else:
            b = b0
        if i >= tstar: # we discard the first 100 iterations
            sample.append(b)
        b0 = b
    print(f'Acceptance ratio: {(a*100)/t}% \n')
    metropolis = np.array(sample)
    for i in range(np.shape(x)[1]):
        print(f'Convergence of averages (sample mean - true value) running the chain for {t} iterations: {np.mean(metropolis[:,i]) - true_beta[i]} \n')
        plt.plot(np.arange(t-100), metropolis[:,i])
        plt.axhline(y=true_beta[i], color='r', linestyle='-')
        plt.title(f'Trace plot for coefficient beta {i} using MH with {prior} prior') 
        plt.show()
    return metropolis

# Gibbs Sampler with one long chain
def gibbs_sampler(X, Y, beta, true_beta, t = 300, *, prior = 'flat', muprior = 0, V = 0):
    
    def fullcond_b(Z, V = V):
        
        if prior == 'flat':
            evbeta = la.inv(X.T @ X) @ (X.T @ Z)
            covbeta = la.inv(X.T @ X)
            return np.random.multivariate_normal(evbeta, covbeta)
        elif prior == 'conjugate': 
            V = la.inv(V)
            evbeta = la.inv(V + X.T @ X) @ (V @ muprior + X.T @ Z)
            covbeta = la.inv(V + X.T @ X)
            return np.random.multivariate_normal(evbeta, covbeta)

    def Z_builder(X, Y, beta):
        Z = []
        for i in range(len(Y)):
            mu, sigma = (X[i] @ beta), 1 # Mean and variance of the full conditional of Z
            if Y[i] == 1:
                zi = stats.truncnorm.rvs((0-mu),100, mu, sigma) # Full conditional of Z[i] truncated at left if Y[i] = 1
                Z.append(zi)
            else:
                zi = stats.truncnorm.rvs(-100,(0-mu), mu, sigma) # Full conditional of Z[i] truncated at left if Y[i] = 0
                Z.append(zi)
        return np.array(Z)

    Z = Z_builder(X, Y, beta)
    betasample = []
    tstar = 100 # We discard the first 100 iterations from the sample because convergence is not yet reached
    for i in range(t): 
        bi = fullcond_b(Z) # At each iteration we update the full conditional of beta by conditioning on the new Z 
        beta = bi
        Z = Z_builder(X, Y, beta) # We update Z conditioning on the new beta
        if i >= tstar:
            betasample.append(beta)

    gibbs = np.array(betasample)     
    for i in range(np.shape(X)[1]):
        print(f'Convergence of averages (sample mean - true value) running the chain for {t} iterations: {np.mean(gibbs[:,i]) - true_beta[i]}')
        plt.plot(np.arange(t - 100), gibbs[:,i])
        plt.axhline(y=true_beta[i], color='r', linestyle='-')
        plt.title(f'Trace plot for coefficient beta {i} using Gibbs sampling with {prior} prior')
        plt.show()
    
    return gibbs

# Gibbs Sampler with Short Chains
def gibbs_sampler_shortchains(X, Y, beta, nchain, true_beta, t, *, prior = 'flat', muprior = 0, V = 0):
    
    def fullcond_b(Z, V = V):
        
        if prior == 'flat':
            evbeta = la.inv(X.T @ X) @ (X.T @ Z)
            covbeta = la.inv(X.T @ X)
            return np.random.multivariate_normal(evbeta, covbeta)
        elif prior == 'conjugate': 
            V = la.inv(V)
            evbeta = la.inv(V + X.T @ X) @ (V @ muprior + X.T @ Z)
            covbeta = la.inv(V + X.T @ X)
            return np.random.multivariate_normal(evbeta, covbeta)

    def Z_builder(X, Y, beta):
        Z = []
        for i in range(len(Y)):
            mu, sigma = (X[i] @ beta), 1 # Mean and variance of the full conditional of Z
            if Y[i] == 1:
                zi = stats.truncnorm.rvs((0-mu),100, mu, sigma) # Full conditional of Z[i] truncated at left if Y[i] = 1
                Z.append(zi)
            else:
                zi = stats.truncnorm.rvs(-100,(0-mu), mu, sigma) # Full conditional of Z[i] truncated at left if Y[i] = 0
                Z.append(zi)
        return np.array(Z)

    beta_initial = beta
    betasample = []
    t_short = t//nchain
    tstar = 100//nchain # We discard the first 100/(n° of chains) iterations 
    i = 0
    while i < nchain:
        Z = Z_builder(X, Y, beta_initial)
        for j in range(t_short): 
            bi = fullcond_b(Z) # At each iteration we update the full conditional of beta by conditioning on the new Z 
            beta = bi
            Z = Z_builder(X, Y, beta) # We update Z conditioning on the new beta
            if j >= tstar:
                betasample.append(beta)
        i += 1
    gibbs = np.array(betasample)
    
    for i in range(np.shape(X)[1]):
        print(f'Convergence of averages (sample mean - true value) running {nchain} chains for {t//nchain} iterations each: {np.mean(gibbs[:,i]) - true_beta[i]}')
        plt.plot(np.arange(t - 100), gibbs[:,i])
        plt.axhline(y=true_beta[i], color='r', linestyle='-')
        plt.title(f'Trace plot for coefficient beta {i} using Gibbs sampling with {prior} prior')
        plt.show()

    return gibbs