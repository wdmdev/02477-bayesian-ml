import numpy as np
import pylab as plt



def plot_summary(ax, x, s, interval=95, num_samples=100, sample_color='k', sample_alpha=0.4, interval_alpha=0.25, color='r', legend=True, title="", plot_mean=True, plot_median=False, label="", seed=0):
    
    b = 0.5*(100 - interval)
    
    lower = np.percentile(s, b, axis=0).T
    upper = np.percentile(s, 100-b, axis=0).T
    
    if plot_median:
        median = np.percentile(s, [50], axis=0).T
        lab = 'Median'
        if len(label) > 0:
            lab += " %s" % label
        ax.plot(x.ravel(), median, label=lab, color=color, linewidth=4)
        
    if plot_mean:
        mean = np.mean(s, axis=0).T
        lab = 'Mean'
        if len(label) > 0:
            lab += " %s" % label
        ax.plot(x.ravel(), mean, '--', label=lab, color=color, linewidth=4)
    ax.fill_between(x.ravel(), lower.ravel(), upper.ravel(), color=color, alpha=interval_alpha, label='%d%% Interval' % interval)    
    
    if num_samples > 0:
        np.random.seed(seed)
        idx_samples = np.random.choice(range(len(s)), size=num_samples, replace=False)
        ax.plot(x, s[idx_samples, :].T, color=sample_color, alpha=sample_alpha);
    
    if legend:
        ax.legend(loc='best')
        
    if len(title) > 0:
        ax.set_title(title, fontweight='bold')
        

def metropolis(log_joint, num_params, tau, num_iter, x_init=None, seed=None):    
    """ Runs a Metropolis-Hastings sampler 
    
        Arguments:
        log_joint:          function for evaluating the log joint distribution
        num_params:         number of parameters of the joint distribution (integer)
        tau:                variance of Gaussian proposal distribution (positive real)
        num_iter:           number of iterations (interger)
        x_init:             vector of initial values (np.array with shape (num_params) or None)        
        seed:               seed (integer or None)

        returns
        xs                  np.array with MCMC samples (np.array with shape (num_iter+1, num_params))
        accept_rate         acceptance rate (non_negative scalar)
    """ 
        
    if seed is not None:
        np.random.seed(seed)

    if x_init is None:
        x_init = np.zeros((num_params))
    
    # prepare lists 
    xs = [x_init]
    accepts = []
    log_p_x = log_joint(x_init)
    
    for k in range(num_iter):

        # get the last value for x and generate new proposal candidate
        x_cur = xs[-1]
        x_star = x_cur + np.random.normal(0, tau, size=(num_params))
        
        # evaluate the log density for the candidate sample
        log_p_x_star = log_joint(x_star)

        # compute acceptance probability
        log_r = log_p_x_star - log_p_x
        A = min(1, np.exp(log_r))
        
        # accept new candidate with probability A
        if np.random.uniform() < A:
            x_next = x_star
            log_p_x = log_p_x_star
            accepts.append(1)
        else:
            x_next = x_cur
            accepts.append(0)

        xs.append(x_next)
        
        
    xs = np.stack(xs)
    return xs.squeeze(), np.mean(accepts)


# implementation borrow from
# from https://github.com/jwalton3141/jwalton3141.github.io/blob/master/assets/posts/ESS/rwmh.py

def gelman_rubin(x):
    """ Estimate the marginal posterior variance. Vectorised implementation. """
    m_chains, n_iters = x.shape

    # Calculate between-chain variance
    B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains*(n_iters - 1))

    # (over) estimate of variance
    s2 = W * (n_iters - 1) / n_iters + B_over_n

    return s2

def compute_effective_sample_size(x):
    """ Compute the effective sample size of estimand of interest. Vectorised implementation. """
    m_chains, n_iters = x.shape

    variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    post_var = gelman_rubin(x)

    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0

        t += 1

    return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))

def compute_Rhat(chains):

    # get dimensions
    num_chains, N = chains.shape
    half_N = int(0.5*N)
    
    # split chains
    sub_chains = []
    for idx_chain in range(num_chains):
        sub_chains.append(chains[idx_chain, :half_N])
        sub_chains.append(chains[idx_chain, half_N:])
        
    M = len(sub_chains)
        
    # compute Rhat statistics
    chain_means = [s.mean() for s in sub_chains]
    global_mean = np.mean(chain_means)
    chain_vars = np.array([1/(N-1)*np.sum((s-m)**2) for (s, m) in zip(sub_chains, chain_means)])

    # compute between chain variance
    B = N/(M-1)*np.sum((chain_means - global_mean)**2)
    
    # within chain variance
    W = np.mean(chain_vars)

    var_estimator = (N-1)/N*W + (1/N)*B
    Rhat = np.sqrt(var_estimator/W)
    
    return Rhat



def combine_chains(chains):
    return chains.flatten()

