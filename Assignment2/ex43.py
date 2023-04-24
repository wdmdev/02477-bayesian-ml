import numpy as np
from autograd.scipy.stats import norm

mu = np.array([3, 3])
Sigma = np.array([[2, 0], [0,3]])**2

phi = lambda x: norm.cdf(x)

def probit_approximation(mu_y, Sigma_y):
    return phi(mu_y.ravel()/np.sqrt(8/np.pi + np.diag(Sigma_y)))

pred_prob = probit_approximation(mu, Sigma)

# print predictive probabilities
print("p(t1 = 1 | t, x1) = ", pred_prob[0].round(4))
print("p(t2 = 1 | t, x2) = ", pred_prob[1].round(4))