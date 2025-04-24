import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats

def fit_mixture_vonmises(series: np.array, mu: np.array, start_pi: np.array, start_kappa: np.array, n: int=2, threshold: float=1e-3):
    
    """
    Find the parameters of a mixture of 2 von Mises distributions, using an EM algorithm (where the means are known)
    
    Input | Type | Details
    -- | -- | --
    series | a 1D numpy array | represent the stochastic periodic process
    mu | a 1D numpy array | the means for the distributions 
    start_pi | a 1D numpy array | the starting weights for the two distributions
    start_kappa | a 1D numpy array | the starting kappas for the two distributions
    n | an int | the number of von Mises distributions in the mixture
    threshold | a float | correspond to the euclidean distance between the old parameters and the new ones (to decide when to terminate the EM)
    
    Output : a (3 x n) numpy-array, containing the probability amplitude of the distribution, 
    and the kappa parameters on each line.
    """

    # initialise the parameters and the distributions
    pi = start_pi 
    kappa = start_kappa

    # probability density function of each data-point given each set of mu and kappa, multiplied by the probability associated with each distribution 
    density = np.array((pi[0]*stats.vonmises.pdf(mu[0],kappa,series), pi[1]*stats.vonmises.pdf(mu[1],kappa,series))).T
    
    # normalize such that pairs of probabilities capturing whether each data point is drawn from distribution A vs distribution B sum to 1 
    sum_density = np.sum(density, axis=1)
    norm_density = (density.T/sum_density).T 
    
    thresh = 1.0 # start with high value for incremental EM convergence 
    
    # calculate and update the coefficients, untill convergence
    
    while (thresh > threshold):
      
        new_pi = np.mean(norm_density, axis=0) # mean across t vector to find overall probabilities of datapoints being drawn from distribution A vs. distribution B
        
        # Fit kappa
        def neg_log_likelihood(x):
              return -sum(np.log(new_pi[0]*stats.vonmises.pdf(mu[0],x,series)+new_pi[1]*stats.vonmises.pdf(mu[1],x,series)))
        
        res = minimize(neg_log_likelihood, kappa, method='L-BFGS-B', tol=1e-10)

        new_kappa = np.array(res.x)
        
        
        # compute update size
        thresh = np.sum((pi-new_pi)**2) + (kappa-new_kappa)**2 # might need to check this line - kappa isn't being counted twice 
        pi = new_pi
        kappa = new_kappa

        # repeat steps to calculate probability density 
        density = np.array((pi[0]*stats.vonmises.pdf(mu[0],kappa,series), pi[1]*stats.vonmises.pdf(mu[1],kappa,series))).T
        sum_density = np.sum(density, axis=1)
        norm_density = (density.T/sum_density).T
        

    # get log liklihood
        
    loglik = -sum(np.log(new_pi[0]*stats.vonmises.pdf(mu[0],kappa,series)+new_pi[1]*stats.vonmises.pdf(mu[1],kappa,series)))
    
    res = np.array([pi, mu, np.squeeze(np.array((kappa,kappa)))])

    return res, loglik

def iter_fit_mixture_vonmises(data: np.array, mu: np.array):
    """
    Fit a mixture of von Mises distributions over a range of initial parameters.
    
    Parameters
    ----------
    data : np.array
        The input data (time series or circular data).
    mu : np.array
        The means for the von Mises distributions.
    
    Returns
    -------
    fit_params : np.array
        Best-fitting parameters (pi, mu, kappa).
    win_llik : float
        Log-likelihood of the best fit.
    init_params : np.array
        Initial parameters corresponding to the best fit.
    """
    
    # Initial sets of pi and kappa values to try
    pis = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], 
           [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
    kappas = [1, 2.5, 5, 10, 15, 20]
    
    # Lists to store all results
    all_params = []
    all_llik = []
    all_initparams = []
        
    # Iterate over initial parameter sets
    for pi in pis:
        
        for kappa in kappas:
            
            init_params = np.array([pi, [kappa, kappa]])  # Stack initial params
            all_initparams.append(init_params)
                        
            # Fit the mixture model using current initial parameters
            param_fit, llik = fit_mixture_vonmises(series=data, mu=mu, start_pi=pi, start_kappa=kappa, n=2)

            # Store results
            all_params.append(param_fit)
            all_llik.append(llik)
                    
    # Find the best fit based on the log-likelihood
    win_idx = np.argmin(all_llik)
    fit_params = all_params[win_idx]
    win_llik = all_llik[win_idx]
    best_init_params = all_initparams[win_idx]
    
    return fit_params, win_llik, best_init_params


def neg_log_likelihood_vm(params, data, mu):
    """
    Calculate negative log likelihood for von Mises distribution.
    
    Parameters
    ----------
    params : array-like
        Parameters for the von Mises distribution. params[0] is kappa.
    data : array-like
        Data points to calculate likelihood for
    mu : float
        Location parameter (mean direction) of the von Mises distribution
        
    Returns
    -------
    float
        Negative log likelihood value. Returns inf if kappa <= 0.
    """
    kappa = params[0]
    if kappa <= 0:  # Ensure kappa is positive
        return np.inf
    nll = -np.sum(stats.vonmises.logpdf(data, kappa, loc=mu))
    return nll

def neg_log_likelihood_weighted(params, data, mu1, mu2):
    """
    Calculate negative log likelihood for weighted mixture of two von Mises distributions.
    
    Parameters
    ----------
    params : array-like
        Parameters for the mixture model. params[0] is kappa, params[1] is weight.
    data : array-like
        Data points to calculate likelihood for
    mu1 : float
        Location parameter of first von Mises component
    mu2 : float
        Location parameter of second von Mises component
        
    Returns
    -------
    float
        Negative log likelihood value. Returns inf if kappa <= 0.
    """
    kappa = params[0]
    w = params[1]
    mu = (w*mu1) + ((1-w)*mu2)
    if kappa <= 0:  # Ensure kappa is positive
        return np.inf
    nll = -np.sum(stats.vonmises.logpdf(data, kappa, loc=mu))
    return nll

def fit_mixture_model(sample, mu_A, mu_B):
    """Fit mixture of von mises distributions to data sample"""
    rule_use, loglik, init_params = iter_fit_mixture_vonmises(sample, mu=[mu_B, mu_A])
    return {
        'A_weight': rule_use[0,1],
        'kappa': rule_use[2,0]
    }


def compare_models(sample, mu_A, mu_B):
    """Compare models and return comparison metrics"""
    A_LL, B_LL = compare_mus(mu_A, mu_B, sample)
    return {
        'A_LL': A_LL,
        'B_LL': B_LL
    }

def compare_mus(mu1, mu2, data):
    
    """
    Compare two von Mises distributions centered at different means by fitting their concentration parameters 
    (kappa) using maximum likelihood estimation and calculating their log likelihoods.
    
    Parameters
    ----------
    mu1 : float
        The mean of the first von Mises distribution.
    mu2 : float
        The mean of the second von Mises distribution.
    data : np.array
        The input circular data to be fit.
    
    Returns
    -------
    log_likelihood_1 : float
        The total log likelihood for the first von Mises distribution (mu1).
    log_likelihood_2 : float
        The total log likelihood for the second von Mises distribution (mu2).
    """

    # Step 1: Fit kappa for each model using MLE (minimizing negative log likelihood)
    result_1 = minimize(neg_log_likelihood_vm, x0=[1], args=(data, mu1), bounds=[(0, None)])
    result_2 = minimize(neg_log_likelihood_vm, x0=[1], args=(data, mu2), bounds=[(0, None)])

    # Extract kappa estimates
    kappa_1 = result_1.x[0]
    kappa_2 = result_2.x[0]
    
    # Compute total log likelihood for model 1
    log_likelihood_1 = np.sum(stats.vonmises.logpdf(data, kappa_1, loc=mu1))
 
    # Compute total log likelihood for model 2
    log_likelihood_2 = np.sum(stats.vonmises.logpdf(data, kappa_2, loc=mu2))
    
    return log_likelihood_1, log_likelihood_2