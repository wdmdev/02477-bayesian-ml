import autograd.numpy as np
import pylab as plt

from scipy.optimize import minimize
from autograd import value_and_grad
from autograd import hessian

def create_linear_regression_data(D, N, kappa2=10, sigma2=20, seed=123):
    np.random.seed(seed)
    
    # create synthetic data
    w_true = np.random.normal(0, np.sqrt(kappa2), size=D)
    X = np.column_stack((np.ones(N), np.random.normal(0, 1, size=(N, D-1))))
    t = X@w_true + np.random.normal(0, np.sqrt(sigma2), size=(N))    
    
    # compute exact solution (using code from exercise 2)
    m_true, S_true = compute_posterior_w(X, t, 1/kappa2, 1/sigma2) 
    exact_marginal = marginal_likelihood(X, t, 1/kappa2, 1/sigma2)
    
    return X, t, w_true, m_true, S_true, exact_marginal



def compute_posterior_w(Phi, t, alpha, beta):
    """ 
    Computes posterior p(w|t) of a linear Gaussian system
    
     Arguments:
         Phi:    NxM matrix of N observations with M features
         t:      Nx1 vector of N observations
         alpha:  real value - hyperparameter of the prior
         beta:   real value - hyperparameter of the likelihood
         
     Returns:
         m:      Mx1 vector for the posterior mean of w
         S:      MxM matrix for the posterior covariance of w
    """
    
    N, M = Phi.shape
    
    # prior precision
    S0 = alpha*np.identity(M)
    
    A = alpha*np.identity(M) + beta*Phi.T@Phi
    
    # compute mean and covariance
    m = beta*np.linalg.solve(A, Phi.T)@t
    S = np.linalg.inv(A)

    return m, S



def marginal_likelihood(Phi, t, alpha, beta):
    """ Computes marginal likelihood of a linear Gaussian system """
    
    N, M = Phi.shape
    m, S = compute_posterior_w(Phi, t, alpha, beta)

    Em = beta/2*np.sum((t - Phi@m)**2) + alpha/2*np.sum(m**2)
    A = alpha*np.identity(M) + beta*Phi.T@Phi

    # marginal likelihood 
    log_Z = M/2*np.log(alpha) + N/2*np.log(beta) - Em - 0.5*np.linalg.slogdet(A)[1] - N/2*np.log(2*np.pi)
    
    return log_Z
    

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
    ax.fill_between(x.ravel(), lower.ravel(), upper.ravel(), color=color, alpha=interval_alpha)    
    
    if num_samples > 0:
        np.random.seed(seed)
        idx_samples = np.random.choice(range(len(s)), size=num_samples, replace=False)
        ax.plot(x, s[idx_samples, :].T, color=sample_color, alpha=sample_alpha);
    
    if legend:
        ax.legend(loc='best')
        
    if len(title) > 0:
        ax.title(title, fontweight='bold')
        
def plot_predictions(ax, x, s, num_samples=100, sample_color='k', sample_alpha=0.4, color='r', legend=False, plot_median=False, plot_mean=True, seed=123, title=''):

    plot_summary(ax, x, s, color=color, interval=99, num_samples=0, interval_alpha=0.25, plot_mean=False, plot_median=False, legend=legend, seed=seed)
    plot_summary(ax, x, s, color=color, interval=95, num_samples=0, sample_alpha=0.1, interval_alpha=0.35, plot_mean=False, plot_median=False, legend=legend, seed=seed)
    plot_summary(ax, x, s, color=color, interval=75, interval_alpha=0.6, num_samples=num_samples, sample_alpha=sample_alpha, plot_mean=False, plot_median=False, legend=legend, seed=seed, sample_color=sample_color)
    
    if plot_median:
        median = np.percentile(s, [50], axis=0).T
        ax.plot(x.ravel(), median, label='Median', color='k', linewidth=4, alpha=0.7)
        
    if plot_mean:
        mean = np.mean(s, axis=0).T
        ax.plot(x.ravel(), mean, '-', label='Mean', color='k', linewidth=4, alpha=0.7)
        
    if title:
        ax.set_title(title, fontweight='bold')
        
        
        
class GradientAscentOptimizer(object):
    
    def __init__(self, num_params, initial_param, step_size):
        self.num_params = num_params
        self.params = initial_param
        self.step_size = step_size
        
    def step(self, gradient):
        self.params = self.params + self.step_size*gradient
        return self.params
        
class StochasticGradientAscentOptimizer(object):
    
    def __init__(self, num_params, initial_param, step_size, tau=0.6, delay=1):
        self.num_params = num_params
        self.params = initial_param
        self.step_size = step_size
        self.tau = tau
        self.delay = delay
        self.itt = 0
        
    def step(self, gradient):
        rho = (self.itt + self.delay)**(-self.tau)
        self.params = self.params + rho*self.step_size*gradient
        self.itt = self.itt + 1
        return self.params

class AdamOptimizer(object):
    
    def __init__(self, num_params, initial_param, step_size):
        self.num_params = num_params
        self.params = initial_param
        self.step_size = step_size
        self.itt = 0
        
        # adam stuff
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8
        
        self.m = np.zeros(self.num_params)
        self.v = np.zeros(self.num_params)
        
    def step(self, gradient):        
        self.m = (1 - self.b1) * gradient + self.b1*self.m
        self.v = (1 - self.b2) * gradient**2 + self.b2*self.v
        mhat = self.m/(1 - self.b1**(self.itt+1))
        vhat = self.v/(1 - self.b2**(self.itt+1))
        self.params = self.params+ self.step_size*mhat/(np.sqrt(vhat)+1e-8)
        self.itt = self.itt + 1
        return self.params
    
    
       
def laplace_approximation(target, w0):
    """ Computes the Laplace approximation of target density """
    
    obj = lambda w: -np.log(target(w))
    
    result = minimize(value_and_grad(obj), w0, jac=True)
            
    if result.success:
        m = result.x
        sigma2_inv = hessian(obj)(m)
        sigma2 = 1/sigma2_inv
        return m[0], sigma2[0,0]
    else:
        print('Optimization failed!')
        return None, None
