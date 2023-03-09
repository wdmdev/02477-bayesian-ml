import autograd.numpy as np
import pylab as plt

from scipy.optimize import minimize
from autograd import value_and_grad
from autograd import grad
from autograd.scipy.stats import norm
from autograd import hessian
from autograd.misc.optimizers import adam
from autograd.misc.flatten import flatten


def log_npdf(x, m, v):
    return -0.5*(x-m)**2/v - 0.5*np.log(2*np.pi*v)

# convert from class label to one-hot encoding
def to_onehot(t, num_classes):
    return np.column_stack([1.0*(t==value) for value in np.arange(num_classes)])

# softmax transformation
def softmax(a):
    exp_a = np.exp(a)
    return exp_a/np.sum(exp_a, 1)[:, None]


class BayesianLinearSoftmax(object):
    """ Bayesian linear softmax classifier with i.i.d. Gaussian priors """
    
    def __init__(self, Phi, t, alpha=1., include_intercept=True):
        
        # data
        self.Phi, self.t  = Phi, t
        self.N, self.D = self.Phi.shape
        
        # num classes
        self.num_classes = len(np.unique(t))
        self.num_params = self.num_classes * self.D
            
        # one-hot encoding
        self.t_onehot = to_onehot(self.t, self.num_classes)
        
        # set prior
        self.alpha = alpha
        self.log_prior = lambda w: np.sum(log_npdf(w, 0, 1./alpha))
        
        # fit
        self.compute_laplace_approximation()
        
    def log_likelihood(self, Phi, t, w):
        # Phi: Input features (N x D)
        # t:   targets (N x 1)
        # w:   flattened parameter vector
        
        # get weights w_i for each latent function y_i
        w_list = np.split(w, self.num_classes)
        
        # compute values for each latent function
        y_all = np.column_stack([self.Phi@w for w in w_list])
        
        # normalize using softmax
        p_all = softmax(y_all)
        
        # evaluate and return value of log likelihood
        return np.sum(self.t_onehot*np.log(p_all))
        
    def log_joint(self, w):
        return self.log_prior(w) + self.log_likelihood(self.Phi, self.t, w)
    
    def compute_laplace_approximation(self):

        w_init = np.zeros(self.num_params)
        cost_fun = lambda w: -self.log_joint(w)
        result = minimize(value_and_grad(cost_fun), w_init, jac=True)

        if result.success:
            w_MAP = result.x
            self.m = w_MAP[:, None]    
            self.A = hessian(cost_fun)(w_MAP)
            self.S = np.linalg.inv(self.A)
            return self.m, self.S
        else:
            print('Warning optimization failed')
            return None, None
    
    def compute_posterior_y(self, Phi):
        """ computes the posterior distribution of y_i(x, w) = w_i^T phi(x) for all classes """
        
        # get relevant part for each of the K functions
        m_parts = np.split(self.m, self.num_classes)
        S_parts = [self.S[i*self.D:(i+1)*self.D, i*self.D:(i+1)*self.D] for i in range(self.num_classes)]
    
        # compute mean and variance for each function
        mu_y_all_classes = np.squeeze(np.stack([Phi@m_parts[i] for i in range(self.num_classes)], axis=1))
        var_y_all_classes = np.squeeze(np.stack([np.diag(Phi@S_parts[i]@Phi.T) for i in range(self.num_classes)], axis=1))
            
        return mu_y_all_classes, var_y_all_classes
        
    
    def compute_predictive_prob(self, Phi, num_samples=500):
        
        # generate samples
        w_samples = np.random.multivariate_normal(self.m.ravel(), self.S, size=num_samples)
        
        # split into samples for individual latent functions
        w_samples_i = np.split(w_samples, self.num_classes, axis=1)

        # compute values for all K linear functions (dim: num_points x num_classes x num_samples)
        y_all_samples = np.stack([Phi@w.T for w in w_samples_i], axis=1)
        
        # compute softmax for all individual samples
        p_all_samples = softmax(y_all_samples)
        
        # compute mean and return 
        p_all = p_all_samples.mean(2)
        
        return p_all


def PCA_dim_reduction(Xtrain, Xtest, num_components):

    
    N = len(Xtrain)
    
    # Center data
    Xm = Xtrain.mean(0)
    Xc_train = Xtrain - Xm
    Xc_test = Xtest - Xm

    # reduce dimensionality using principal component analysis (PCA) via SVD
    U, s, V = np.linalg.svd(Xc_train)

    # get eigenvectors corresponding to the two largest eigenvalues
    eigen_vecs = V[:num_components, :]
    eigen_vals = s[:num_components]

    # set-up projection matrix
    Pmat = eigen_vecs.T*(np.sqrt(N)/eigen_vals)

    # project and standize
    Ztrain = Xc_train@Pmat
    Ztest = Xc_test@Pmat 
    return Ztrain, Ztest

def visualize_utility(ax, U, labels=None):
    
    num_classes = len(U)
    
    ax.imshow(U, cmap=plt.cm.Greys_r, alpha=0.5)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
    ax.grid(False)
    
    for (j,i), val in np.ndenumerate(U):
        ax.text(i,j, val, ha='center', va='center', fontsize=16)
    ax.set_title('Utility matrix', fontweight='bold')
