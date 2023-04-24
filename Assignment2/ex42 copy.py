import numpy as np

sigmoid = lambda x: 1./(1+np.exp(-x))

def generate_samples(mean, K, M, jitter=1e-8):
    """ returns M samples from a zero-mean Gaussian process with kernel matrix K
    
    arguments:
    K      -- NxN kernel matrix
    M      -- number of samples (scalar)
    jitter -- scalar
    returns NxM matrix
    """
    
    L = np.linalg.cholesky(K + jitter*np.identity(len(K)))
    zs = np.random.normal(0, 1, size=(len(K), M))
    fs = mean + np.dot(L, zs)
    return fs

def compute_predictive_prob_MC(mu_y, Sigma_y, sample_size=2000):
    """
        The function computes p(t^* = 1|t, x^*) using Monte Carlo sampling  as in eq. (2).
        The function also returns the samples generated in the process for plotting purposes

        arguments:
        mu_y             -- N x 1 vector
        Sigma_y          -- N x N matrix
        sample_size      -- positive integer

        returns:
        p                -- N   vector
        y_samples        -- sample_size x N matrix
        sigma_samples    -- sample_size x N matrix

    """

    # generate samples from y ~ N(mu, Sigma)
    y_samples = generate_samples(mu_y, Sigma_y, sample_size).T 

    # apply inverse link function (elementwise)
    sigma_samples = sigmoid(y_samples)

    # return MC estimate, samples of y and sigma(y)
    return np.mean(sigma_samples, axis=0), y_samples, sigma_samples

MCSE = lambda samples: np.std(samples, axis = 0) / np.sqrt(len(samples))

mu = np.array([3, 3])
Sigma = np.array([[2, 0], [0,3]])**2
S = int(9e6)

pred_prob, y_samples, p_samples = compute_predictive_prob_MC(mu, Sigma, S)
mcse = MCSE(y_samples)


# print predictive probabilities
print("p(t1 = 1 | t, x1) = ", pred_prob[0].round(4))
print("p(t2 = 1 | t, x2) = ", pred_prob[1].round(4))
# print mcse
format_mcse = lambda x: (x*1e3).round(4)
print("MCSE 1 = ", (mcse[0]*1e3).round(4), "e-3")
print("MCSE 2 = ", (mcse[1]*1e3).round(4), "e-3")