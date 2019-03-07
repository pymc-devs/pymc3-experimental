from pymc3.step_methods.arraystep import Competence
from pymc3.step_methods.sgmcmc import BaseStochasticGradient
from pymc3.vartypes import continuous_types
import theano.tensor as tt
import theano
import numpy as np

__all__ = ['SGFS', 'CSG']


class SGFS(BaseStochasticGradient):
    R"""
    StochasticGradientFisherScoring

    Parameters
    ----------
    vars : list
        model variables
    B : np.array
        the pre-conditioner matrix for the fisher scoring step
    step_size_decay : int
        Step size decay rate. Every `step_size_decay` iteration the step size reduce
        to the half of the previous step size

    References
    ----------
    -   Bayesian Posterior Sampling via Stochastic Gradient Fisher Scoring
        Implements Algorithm 1 from the publication http://people.ee.duke.edu/%7Elcarin/782.pdf
    """
    name = 'stochastic_gradient_fisher_scoring'

    def __init__(self, vars=None, B=None, step_size_decay=100, **kwargs):
        """
        Parameters
        ----------
        vars : list
            Theano variables, default continuous vars
        B : np.array
            Symmetric positive Semi-definite Matrix
        kwargs: passed to BaseHMC
        """
        self.B = B
        self.step_size_decay = step_size_decay
        super().__init__(vars, **kwargs)

    def _initialize_values(self):
        # Init avg_I
        self.avg_I = theano.shared(
            np.zeros((self.q_size, self.q_size)), name='avg_I')
        self.t = theano.shared(1, name='t')
        # 2. Set gamma
        self.gamma = (self.batch_size + self.total_size) / (self.total_size)

        self.training_fn = self.mk_training_fn()

    def mk_training_fn(self):

        n = self.batch_size
        N = self.total_size
        q_size = self.q_size
        B = self.B
        gamma = self.gamma
        avg_I = self.avg_I
        t = self.t
        updates = self.updates
        epsilon = self.step_size / pow(2.0, t // self.step_size_decay)
        random = self.random
        inarray = self.inarray
        gt, dlog_prior = self.dlogp_elemwise, self.dlog_prior

        # 5. Calculate mean dlogp
        avg_gt = gt.mean(axis=0)

        # 6. Calculate approximate Fisher Score
        gt_diff = (gt - avg_gt)

        V = (1. / (n - 1)) * tt.dot(gt_diff.T, gt_diff)

        # 7. Update moving average
        I_t = (1. - 1. / t) * avg_I + (1. / t) * V

        if B is None:
            # if B is not specified
            # B \propto I_t as given in
            # http://www.ics.uci.edu/~welling/publications/papers/SGFS_v10_final.pdf
            # after iterating over the data few times to get a good approximation of I_N
            B = tt.switch(t <= int(N / n) * 50, tt.eye(q_size), gamma * I_t)

        # 8. Noise Term
        # The noise term is sampled from a normal distribution
        # of mean 0 and std_dev = sqrt(4B/step_size)
        # In order to generate the noise term, a standard
        # normal dist. is scaled with 2B_ch/sqrt(step_size)
        # where B_ch is cholesky decomposition of B
        # i.e. B = dot(B_ch, B_ch^T)
        B_ch = tt.slinalg.cholesky(B)
        noise_term = tt.dot((2.*B_ch)/tt.sqrt(epsilon), \
                random.normal((q_size,), dtype=theano.config.floatX))
        # 9.
        # Inv. Fisher Cov. Matrix
        cov_mat = (gamma * I_t * N) + ((4. / epsilon) * B)
        inv_cov_mat = tt.nlinalg.matrix_inverse(cov_mat)
        # Noise Coefficient
        noise_coeff = (dlog_prior + (N * avg_gt) + noise_term)
        dq = 2 * tt.dot(inv_cov_mat, noise_coeff)

        updates.update({avg_I: I_t, t: t + 1})

        f = theano.function(
            outputs=dq,
            inputs=inarray,
            updates=updates,
            allow_input_downcast=True)

        return f

    @staticmethod
    def competence(var, has_grad):
        if var.dtype in continuous_types and has_grad:
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE


class CSG(BaseStochasticGradient):
    R"""
    CSG: ConstantStochasticGradient

    It is an approximate stochastic variational inference algorithm
    while SGFS and many other MCMC techniques provably
    converge towards the exact posterior. The referenced paper
    discusses a proof for the optimal preconditioning matrix
    based on variational inference, so there is no parameter tuning required
    like in the case of 'B' matrix used for preconditioning in SGFS.
    Take a look at this example notebook
    https://github.com/pymc-devs/pymc3/tree/master/docs/source/notebooks/constant_stochastic_gradient.ipynb

    Parameters
    ----------
    vars : list
        model variables

    References
    ----------
    -   Stochastic Gradient Descent as Approximate Bayesian Inference
        https://arxiv.org/pdf/1704.04289v1.pdf
    """
    name = 'constant_stochastic_gradient'

    def __init__(self, vars=None, **kwargs):
        """
        Parameters
        ----------
        vars : list
            Theano variables, default continuous vars
        kwargs: passed to BaseHMC
        """
        super().__init__(vars, **kwargs)

    def _initialize_values(self):
        # Init avg_C: Noise Covariance Moving Average
        self.avg_C = theano.shared(
            np.zeros((self.q_size, self.q_size)), name='avg_C')
        self.t = theano.shared(1, name='t')
        # Init training fn
        self.training_fn = self.mk_training_fn()

    def mk_training_fn(self):
        """The Constant Stochastic Gradient Step Fn with Optimal Preconditioning Matrix"""
        q_size = self.q_size
        avg_C = self.avg_C
        t = self.t
        updates = self.updates
        # Trying to stick to variables names as given in the publication
        # https://arxiv.org/pdf/1704.04289v1.pdf
        S = self.batch_size
        N = self.total_size

        # inputs
        random = self.random
        inarray = self.inarray

        # gradient of log likelihood
        gt = -1 * (1. / S) * (self.dlogp_elemwise.sum(axis=0) +
                              (S / N) * self.dlog_prior)

        # update moving average of Noise Covariance
        gt_diff = (self.dlogp_elemwise - self.dlogp_elemwise.mean(axis=0))
        V = (1. / (S - 1)) * theano.dot(gt_diff.T, gt_diff)
        C_t = (1. - 1. / t) * avg_C + (1. / t) * V
        # BB^T = C
        B = tt.switch(t < 0, tt.eye(q_size), tt.slinalg.cholesky(C_t))
        # Optimal Preconditioning Matrix
        H = (2. * S / N) * tt.nlinalg.matrix_inverse(C_t)
        # step value on the log likelihood gradient preconditioned with H
        step = -1 * theano.dot(H, gt.dimshuffle([0, 'x']))

        # sample gaussian noise dW
        dW = random.normal(
            (q_size, 1), dtype=theano.config.floatX, avg=0.0, std=1.0)
        # noise term is inversely proportional to batch size
        noise_term = (1. / np.sqrt(S)) * theano.dot(H, theano.dot(B, dW))
        # step + noise term
        dq = (step + noise_term).flatten()

        # update time and avg_C
        updates.update({avg_C: C_t, t: t + 1})

        f = theano.function(
            outputs=dq,
            inputs=inarray,
            updates=updates,
            allow_input_downcast=True)

        return f

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE
