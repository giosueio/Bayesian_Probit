# Bayesian estimate of Probit coefficients 
##### _Last update of scripts: December 2019 (late upload on GitHub)_

Implementation of a Metropolis Hasting and a Gibbs Sampler with auxiliary variables to sample from the posterior distribution of the coefficients of a probit model for a binary response variable, as proposed in Albert, J. H., & Chib, S. (1993).
The model is implemented both with conjugate and with flat priors.


## Tutorial
The module has three functions
* `mh_norm` performs Metropolis-Hastings iterations
* `gibbs_sampler` performs Gibbs Sampling iterations
* `gibbs_sampler_shortchains` performs Gibbs Sampling with an alternative implementation (the chain restart at the initial value after a given number of iterations)



When called, each of the functions produces a simulated sample from the posterior, and returns a trace plot.
  
  
  

<img src="https://github.com/giosueio/Bayesian_Probit/blob/master/convergence.png" width="30%" length="30%">

## Credits
Project by Giosuè Migliorini ([giosue.migliorini@gmail.com](mailto:giosue.migliorini@gmail.com)) (code) and Andrea Raminelli (report) as part of the course [20592 - STATISTICS AND PROBABILITY](https://didattica.unibocconi.eu/ts/tsn_anteprima.php?cod_ins=20592&anno=2022&IdPag=) at Bocconi University, taught by professors S.Petrone and R.Graziani.

## References
* Albert, J. H., & Chib, S. (1993). Bayesian Analysis of Binary and Polychotomous Response Data. Journal of the American Statistical Association, 88(422), 669–679. https://doi.org/10.2307/2290350
