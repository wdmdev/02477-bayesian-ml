import autograd.numpy as np
import pylab as plt
import seaborn as snb

from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

def plot_with_uncertainty(ax, Xp, mu, Sigma, sigma, color='r', color_samples='b', title="", num_samples=0):
    
    mean, std = mu.ravel(), np.sqrt(np.diag(Sigma) + sigma**2)

    
    # plot distribution
    ax.plot(Xp, mean, color=color, label='Mean')
    ax.plot(Xp, mean + 2*std, color=color, linestyle='--')
    ax.plot(Xp, mean - 2*std, color=color, linestyle='--')
    ax.fill_between(Xp.ravel(), mean - 2*std, mean + 2*std, color=color, alpha=0.25, label='95% interval')
    
    # generate samples
    if num_samples > 0:
        fs = generate_samples(mu, Sigma, num_samples)
        ax.plot(Xp, fs[:,0], color=color_samples, alpha=.25, label="$y(x)$ samples")
        ax.plot(Xp, fs[:, 1:], color=color_samples, alpha=.25)
    
    ax.set_title(title)
    

def generate_samples(mean, K, M, jitter=1e-8):
    """ returns M samples from a zero-mean Gaussian process with kernel matrix K
    
    arguments:
    K      -- NxN kernel matrix
    M      -- number of samples (scalar)
    jitter -- scalar
    returns NxM matrix
    """
    
    jitter = 1e-8
    L = np.linalg.cholesky(K + jitter*np.identity(len(K)))
    zs = np.random.normal(0, 1, size=(len(K), M))
    fs = mean + np.dot(L, zs)
    return fs
