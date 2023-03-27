import numpy as np
from time import time

from scipy.special import psi
from scipy.special import logsumexp
from scipy.special import gammaln

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms




########################################################################################################################
# Helper functions
########################################################################################################################
def log_B(W, nu):
    D = len(W)
    return  -0.5*nu*np.linalg.slogdet(W)[1] - (nu*D/2)*np.log(2) - (D*(D-1)/4)*np.log(np.pi) -np.sum(gammaln((nu + 1 -  np.arange(1, D+1))/2))

def log_C(alpha):
    K = len(alpha)
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

def mvtGammaln(n, alpha):
    return ((n*(n-1))/4)*np.log(np.pi)+np.sum(gammaln(alpha+0.5*(1-np.arange(1,n+1))));
    
def wishart_log_normalizer(W, nu, logdetW):
    D = len(W)
    g = ((D*(D-1))/4)*np.log(np.pi)+np.sum(gammaln(nu/2+0.5*(1-np.arange(1,D+1))));
    return -(nu/2)*logdetW -(nu*D/2)*np.log(2) - mvtGammaln(D,nu/2);

def wishart_expected_logdet(W, nu, logdetW):
    D = len(W)
    return np.sum(psi(1/2*(nu + 1 - np.arange(1, D+1)))) + D*np.log(2)  + logdetW;


def wishart_entropy(W, nu, logdetW):
    D = len(W)  
    return -wishart_log_normalizer(W, nu, logdetW) - (nu-D-1)/2*wishart_expected_logdet(W, nu, logdetW) + nu*D/2;

def log_mv_student_t(x, mu, Lambda, nu):
    D = len(Lambda)    
    d = x - mu
    d2 = np.sum((d@Lambda)*d, axis=1)
    return gammaln(D/2 + nu/2) - gammaln(nu/2) + 0.5*np.linalg.slogdet(Lambda)[1] - 0.5*D*np.log(np.pi*nu) + (-D/2 -nu/2)*np.log(1 + d2/nu)
    
    
########################################################################################################################
# Basic implementation of Variational GMM from Bishop's book
########################################################################################################################    
class VariationalGMM(object):
    
    def __init__(self, D, K, alpha0=0.5, beta0=1., seed=0):
        """ Parameters:
            ---------------------------------
            D:      Number of dimensions
            K:      Number of components
            alpha0: Hyperparameter for Dirichlet distribution
            beta0:  Hyperparameter for Gaussian-Wishart prior
            seed:   Seed
            
            The hyperparameters m0 and W0 are assumed to be the zero-vector and the identity matrix, respectively.
            
        """
        
        # dimensions and number of components
        self.D = D
        self.K = K
        
        # prior
        self.m0 = np.zeros(self.D)
        self.W0 = np.identity(D)
        self.W0inv = np.linalg.inv(self.W0)
        
        # hyperparameters
        self.nu0 = D
        self.alpha0 = alpha0*np.ones(self.K)
        self.beta0 = beta0
        
        # convergence stuff
        self.jitter = 1e-10
        self.convergence_tol = 1e-6
        self.converged = False
        self.seed = seed
        
        
        # monitoring and evaluation
        self.num_active_components = None
        self.lowerbound_history = None
        self.pi_history = None
        self.test_lpd_history = None
        self.lowerbound = None
        self.test_lpd = None
        
        
                
    def fit(self, X, Xtest=None, max_itt=500, verbose=False):
        """ Fit variational approximation for Bayesian GMM model """

        np.random.seed(self.seed)
        
        t0 = time()
        
        self.X = X
        self.N = len(self.X)
        
        # Do we have test data?
        if Xtest is not None:
            self.Xtest = Xtest
            Ntest = len(Xtest)
            self.test_lpd = []
            
        # prepare to store ELBOs etc
        self.lowerbound_history = []
        self.test_lpd_history = []
        self.pi_history = []
        self.num_active_components = []
                
        # initialize responsibilities
        self.log_rho = np.random.normal(0, .01, size=(self.N, self.K))
        self.log_r = self.log_rho - logsumexp(self.log_rho, 1)[:, None]
        self.r = np.exp(self.log_r)


        # iterate
        lowerbound_old = -np.Inf
        self.converged = False
        for itt in range(max_itt):
            
            ###############################################################################################
            # Pre-compute relevant statistics given current values
            ###############################################################################################
            
            # compute counts using eq. (10.51)    
            self.Nk = np.sum(self.r, 0) + 1e-10
    
            # eq. (10.52)
            self.Xbar = np.sum(self.r[:, :, None]*X[:, None, :], axis=0)/self.Nk[:, None]

            # eq. (10.53)
            self.S = np.zeros((self.K, self.D, self.D))
            for k in range(self.K):
                Xdiff = X - self.Xbar[k]
                Xdiff2 = self.r[:,k, None, None]*(Xdiff[:, :, None]@Xdiff[:, None, :])
                self.S[k, :, :] =np.sum(Xdiff2, axis=0)/self.Nk[k]
                
            ###############################################################################################
            # Update posterior q(pi) = Dir(pi|alpha_k) using eq. (10.58)
            ###############################################################################################
            self.alpha_k = self.alpha0 + self.Nk
            
            ###############################################################################################
            # Update posterior q(mu_k, Lambda_k) = N(mu_k|m_k, (beta_k Lambda_k)^{-1}) W(Lambda_k|W_k, nu_k)
            ###############################################################################################
            
            # Update beta_k using eq. (10.60)
            self.beta_k = self.beta0 + self.Nk

            # Update m_k using eq. (10.61)
            self.m = np.zeros((self.K, self.D))
            for k in range(self.K):
                self.m[k, :] = (self.beta0*self.m0 + self.Nk[k]*self.Xbar[k])/self.beta_k[k]

            # Update w_k using eq. (10.62)    
            self.Wk_inv = np.zeros((self.K, self.D, self.D))
            self.Wk = np.zeros((self.K, self.D, self.D))
            for k in range(self.K):
                self.Wk_inv[k] = self.W0inv + self.Nk[k]*self.S[k] + (self.beta0*self.Nk[k])/(self.beta0 + self.Nk[k])*np.outer(self.Xbar[k] - self.m0, self.Xbar[k] - self.m0) + self.jitter*np.identity(self.D)
                self.Wk[k] = np.linalg.inv(self.Wk_inv[k])
                
            # Update nu_k using eq. (10.63)
            self.nu_k = self.nu0 + self.Nk
            
            ###############################################################################################
            # Update posterior q(z_n) = Cat(z_n|r_n)
            ###############################################################################################

            # Compute quantity in eq. (10.65)
            self.ln_lam = np.zeros(self.K)
            self.logdetWk = np.zeros(self.K)
            for k in range(self.K):
                self.logdetWk[k] = np.linalg.slogdet(self.Wk[k])[1]
                self.ln_lam[k] +=  np.sum(psi((self.nu_k[k] + 1 - np.arange(1, self.D+1))/2)) + self.D*np.log(2) + self.logdetWk[k]

            # Compute quantity in eq. (10.66)
            self.ln_pi_tilde = psi(self.alpha_k) - psi(np.sum(self.alpha_k))
                
            # Finally, compute parameters of q(z_n) using eq. (10.67)
            self.log_rho = np.zeros((self.N, self.K))
            for k in range(self.K):
                xdiff = self.X - self.m[k]
                d2 = np.sum((xdiff@self.Wk[k])*xdiff, 1)
                self.log_rho[:, k] = self.ln_pi_tilde[k] + 0.5*self.ln_lam[k] -self.D/(2*self.beta_k[k]) -0.5*self.nu_k[k]*d2
                
            # normalize and exponentiate to get a proper distribution
            self.log_r = self.log_rho - logsumexp(self.log_rho, 1)[:, None]
            self.r = np.exp(self.log_r)

            ###############################################################################################
            # Monitor quantities of interest
            ###############################################################################################
            
            # Compute and store posterior mean of mixing weights via .eq. (10.69)
            self.pi = (self.alpha0 + self.Nk)/(np.sum(self.alpha0) + self.N)
            self.pi_history.append(self.pi)
            
            # Estimate number of "active components"
            self.num_active_components.append(np.sum(self.Nk >= 1))    
            
            # compute and store lower bound
            self.compute_lowerbound()
            self.lowerbound_history.append(self.lowerbound)
            
            ###############################################################################################
            # Evaluate test log predictive density if given a test set
            ###############################################################################################
            if Xtest is not None:
                self.test_lpd = self.evaulate_log_predictive(Xtest)
                self.test_lpd_history.append(self.test_lpd)
                
            ###############################################################################################
            # check for convergence
            ###############################################################################################
            if np.abs(lowerbound_old - self.lowerbound_history[-1]) < self.convergence_tol:
                t1 = time()
                print('Converged in %4d iterations for K = %2d with lowerbound = %4.3f. Time = %3.2f' % (itt, self.K, self.lowerbound, t1-t0))
                self.converged = True
                break
            lowerbound_old = self.lowerbound_history[-1]
            
            if verbose:
                if (itt+1) % 100 == 0:
                    print('Itt=%d/%d, lowerbound=%3.2f in %3.2fs' % (itt+1, max_itt, self.lowerbound, time()-t0))
            
            
        if not self.converged:
            print('Warning: Optimization did not converge')
            
        return self
            
        
        

    def compute_lowerbound(self):
        """  evaluate lower bound using eq. (10.70)-(10.77) given current variational parameters """

        t1 = 0
        # eq. (10.71)
        for k in range(self.K):
            t1 += 0.5*self.Nk[k]*(self.ln_lam[k] - self.D/self.beta_k[k] - self.nu_k[k]*np.trace(self.S[k]@self.Wk[k]) -  self.nu_k[k]*(self.Xbar[k] - self.m[k])@self.Wk[k]@(self.Xbar[k] - self.m[k]).T - self.D*np.log(2*np.pi))
            
        # eq. (10.72)
        t2 = np.sum(self.r*self.ln_pi_tilde)

        # eq. (10.73)
        t3 = log_C(self.alpha0) + np.sum((self.alpha0-1)*self.ln_pi_tilde)

        # eq. (10.74)
        t4 = 0
        for k in range(self.K):
            d = self.m[k]-self.m0        
            t4 += 0.5*(self.D*np.log(self.beta0/(2*np.pi)) + self.ln_lam[k] - self.D*self.beta0/self.beta_k[k] - self.beta0*self.nu_k[k]*d.T@self.Wk[k]@d)
        t4 += self.K*log_B(self.W0, self.nu0)  + 0.5*(self.nu0 - self.D - 1)*np.sum(self.ln_lam)
        for k in range(self.K):
            t4 += -0.5*self.nu_k[k]*np.trace(self.W0inv@self.Wk[k])
            
        # eq. (10.75)
        t5 = np.sum(self.r*self.log_r)

        # eq. (10.76)
        t6 = np.sum( (self.alpha_k-1)*self.ln_pi_tilde) + log_C(self.alpha_k)

        # eq. (10.77)                                           
        t7 = 0
        for k in range(self.K):
            t7 += 0.5*self.ln_lam[k] + 0.5*self.D*np.log(self.beta_k[k]/(2*np.pi)) - 0.5*self.D - wishart_entropy(self.Wk[k],self.nu_k[k], self.logdetWk[k])
            
        # sum all contributions (eq. (10.70))
        self.lowerbound = t1 + t2 + t3 + t4 - t5 - t6 - t7
        
    
        
    def compute_component_probs(self, Xp):
        """ Compute posterior component probabilitites P(Zp=k|X, Xp) for each k = 1, ..., K """
        
        # compute precision matrix for each component
        Lk = [(self.nu_k[k] +1 - self.D)*self.beta_k[k]/(1 + self.beta_k[k])*self.Wk[k] for k in range(self.K)]
        
        # compute log mixing weights
        log_weights = np.log(self.alpha_k) - np.log(np.sum(self.alpha_k))
        
        # compute component contributions
        log_component_contributions = []
        for k in range(self.K):
            log_val = log_weights[k] + log_mv_student_t(Xp, self.m[k], Lk[k], self.nu_k[k] + 1 - self.D)
            log_component_contributions.append(log_val)
            
        # normalize in log-space
        log_probs = np.array(log_component_contributions) - logsumexp(log_component_contributions, axis=0)
        
        # exponentiate and return
        return np.exp(log_probs.T)

        
    def evaulate_log_predictive(self, Xp, pointwise=False):
        """ Compute the log predictive densities for test points Xp using log p(Xp|X) in eq. (10.81).
            If pointwise = False, return the sum of log predictive densities for all points in Xp
            """
        
        # compute precision matrix for each component
        Lk = [(self.nu_k[k] +1 - self.D)*self.beta_k[k]/(1 + self.beta_k[k])*self.Wk[k] for k in range(self.K)]
        
        # compute log mixing weights
        log_weights = np.log(self.alpha_k) - np.log(np.sum(self.alpha_k))
        
        # compute component contributions
        log_component_contributions = []
        for k in range(self.K):
            log_val = log_weights[k] + log_mv_student_t(Xp, self.m[k], Lk[k], self.nu_k[k] + 1 - self.D)
            log_component_contributions.append(log_val)

        # compute log predictive densities
        l = logsumexp(log_component_contributions, axis=0)
        
        if not pointwise:
            l = l.sum()
        
        return l
     
    
    
        


def plot_std_dev_contour(ax, m, cov, n_std=3.0, facecolor='none', **kwargs):
    """
    Adapted from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = m[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = m[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def PCA_dim_reduction(X, num_components):

    
    N = len(X)
    
    # Center data
    Xm = X.mean(0)
    Xc = X - Xm

    # reduce dimensionality using principal component analysis (PCA) via SVD
    U, s, V = np.linalg.svd(Xc)

    # get eigenvectors corresponding to the two largest eigenvalues
    eigen_vecs = V[:num_components, :]
    eigen_vals = s[:num_components]

    # set-up projection matrix
    Pmat = eigen_vecs.T*(np.sqrt(N)/eigen_vals)

    # project and standize
    return Xc@Pmat

