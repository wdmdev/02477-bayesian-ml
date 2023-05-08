import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.misc.flatten import flatten
from autograd.scipy.special import logsumexp

def log_npdf(x, m, v):
    return -0.5*(x-m)**2/v - 0.5*np.log(2*np.pi*v)



#############################################################################
# Adapted from
# https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
#############################################################################

class NeuralNetwork(object):

    def __init__(self, X, t, layer_sizes, log_lik_fun, alpha=1., step_size=0.01, num_iters=1000, batch_size=None, seed=0):

        # data
        self.X = X
        self.t = t
        self.N = len(self.X)

        # model and optimization parameters
        self.log_lik_fun = log_lik_fun
        self.layer_sizes = layer_sizes
        self.step_size = step_size
        self.num_iters = num_iters
        self.alpha = alpha
        self.batch_size = batch_size

        self.activation_fun = np.tanh

        # random number genration
        self.seed=seed
        self.rng = np.random.default_rng(seed)
        
        # initialize parameters and optimize
        self.params = self.init_random_params()
        self.optimize()


    def init_random_params(self):
        """Build a list of (weights, biases) tuples,
        one for each layer in the net."""
        return [(np.sqrt(2/n) * self.rng.standard_normal((m, n)),   # weight matrix
                np.sqrt(2/n) * self.rng.standard_normal(n))      # bias vector
                for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]



    def first_layers(self, params, inputs):
        """ implements the map from the input features to the activation of layer L-1 """
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = self.activation_fun(outputs)
        return inputs

    def last_layer(self, last_params, first_layers):
        """ implements the map from the activation of layer L-1 to the output of the network"""
    
        W, b = last_params
        return np.dot(first_layers, W) +b 


    def neural_net_predict(self, params, inputs):
        """Implements a deep neural network for classification.
        params is a list of (weights, bias) tuples.
        inputs is an (N x D) matrix.
        returns logits."""
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = self.activation_fun(outputs)
        return outputs

    def predict(self, inputs):
        return self.neural_net_predict(self.params, inputs)

    def log_prior(self, params):
        # implement a Gaussian prior on the weights
        flattened_params, _ = flatten(params)
        return  np.sum(log_npdf(flattened_params, 0., 1/self.alpha))
    
    def log_likelihood(self, params, compute_full=False):  

        if self.batch_size is None or compute_full:
            y = self.neural_net_predict(params, self.X)
            log_lik = np.sum(self.log_lik_fun(y.reshape((self.N, -1)), self.t))
        else:
            batch_idx = np.random.choice(range(self.N), size=self.batch_size, replace=False)
            X_batch, t_batch = self.X[batch_idx, :], self.t[batch_idx]
            y = self.neural_net_predict(params, X_batch)
            log_lik = self.N/self.batch_size*np.sum(self.log_lik_fun(y.reshape((self.batch_size, -1)), t_batch))
        
        return log_lik

    def log_joint(self, params, compute_full=False):
        return  self.log_prior(params) + self.log_likelihood(params, compute_full)

    def callback(self, params, itt, gradient):
        if (itt +1) % int(0.1*self.num_iters) == 0:
            loss_ = self.log_joint(params, compute_full=True)
            print(f'Itt = {itt+1}/{self.num_iters} ({100*(itt+1)/self.num_iters}%), log joint = {loss_:3.2f}')
            self.loss.append(loss_)
        

    def optimize(self):

        # Define training objective and gradient of objective using autograd.
        def objective(params, iter):
            return -self.log_joint(params)
            
        objective_grad = grad(objective)

        # optimize
        self.loss = []
        self.params = adam(objective_grad, self.params, step_size=self.step_size, num_iters=self.num_iters, callback=self.callback)

        return self

    

