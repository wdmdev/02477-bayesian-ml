import numpy as np

# set given parameters
mu = np.array([3, 3])
Sigma = np.array([[2, 0], [0,3]])**2
# create helper functions
sigmoid = lambda x: 1./(1+np.exp(-x))
MCSE = lambda samples: np.std(samples, axis = 0) / np.sqrt(len(samples))

def MC_estimation(N, mu, sigma, seed = 0):
    # set seed
    np.random.seed(seed)
    # generate samples
    samples = np.random.multivariate_normal(mu, sigma, size = N)
    # calculate MCSE
    mcse = MCSE(samples)
    # compute predictive probabilities
    pred_prob = sigmoid(samples).mean(axis = 0)
    return pred_prob, mcse

# run sampling
pred_prob, mcse = MC_estimation(N = int(9e6), mu = mu, sigma= Sigma)
# print predictive probabilities
print("p(t1 = 1 | t, x1) = ", pred_prob[0].round(4))
print("p(t2 = 1 | t, x2) = ", pred_prob[1].round(4))
# print mcse
format_mcse = lambda x: (x*1e3).round(4)
print("MCSE 1 = ", (mcse[0]*1e3).round(4), "e-3")
print("MCSE 2 = ", (mcse[1]*1e3).round(4), "e-3")