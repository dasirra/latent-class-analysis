import numpy as np
import scipy.stats as stats


class LCA:
    def __init__(self, n_components=2, tol=1e-3, max_iter=100, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter

        # flag to indicate if converged
        self.converged_ = False

        # model parameters
        self.ll_ = [-np.inf]
        self.weight = None
        self.theta = None
        self.responsibility = None

        # bic estimation
        self.bic = None

        # verbose level
        self.verbose = 0

    def _calculate_responsibility(self, data):

        n_rows, n_cols = np.shape(data)
        r_numerator = np.zeros(shape=(n_rows, self.n_components))
        for k in range(self.n_components):
            r_numerator[:, k] = self.weight[k] * np.prod(stats.bernoulli.pmf(
                data, p=self.theta[k]), axis=1)
        r_denominator = np.sum(r_numerator, axis=1)
        return r_numerator / np.tile(r_denominator, (self.n_components, 1)).T

    def _do_e_step(self, data):

        self.responsibility = self._calculate_responsibility(data)

    def _do_m_step(self, data):

        n_rows, n_cols = np.shape(data)

        # pi
        for k in range(self.n_components):
            self.weight[k] = np.sum(self.responsibility[:, k]) / float(n_rows)

        # theta
        for k in range(self.n_components):
            numerator = np.zeros((n_rows, n_cols))
            for n in range(n_rows):
                numerator[n, :] = self.responsibility[n, k] * data[n, :]
            numerator = np.sum(numerator, axis=0)
            denominator = np.sum(self.responsibility[:, k])
            self.theta[k] = numerator / denominator

        # correct numerical issues
        mask = self.theta > 1.0
        self.theta[mask] = 1.0
        mask = self.theta < 0.0
        self.theta[mask] = 0.0

    def fit(self, data):

        # initialization step
        n_rows, n_cols = np.shape(data)
        if n_rows < self.n_components:
            raise ValueError(
                '''
                LCA estimation with {n_components} components, but got only
                {n_rows} samples
                '''.format(n_components=self.n_components, n_rows=n_rows))

        if self.verbose > 0:
            print('EM algorithm started')

        self.weight = stats.dirichlet.rvs(np.ones(shape=self.n_components) / 2, random_state=self.random_state)[0]
        self.theta = stats.dirichlet.rvs(alpha=np.ones(shape=n_cols) / 2,
                                         size=self.n_components,
                                         random_state=self.random_state)

        for i in range(self.max_iter):
            if self.verbose > 0:
                print('\tEM iteration {n_iter}'.format(n_iter=i))

            # E-step
            self._do_e_step(data)

            # M-step
            self._do_m_step(data)

            # Check for convergence
            aux = np.zeros(shape=(n_rows, self.n_components))
            for k in range(self.n_components):
                normal_prob = np.prod(stats.bernoulli.pmf(data, p=self.theta[k]), axis=1)
                aux[:, k] = self.weight[k] * normal_prob
            ll_val = np.sum(np.log(np.sum(aux, axis=1)))
            if np.abs(ll_val - self.ll_[-1]) < self.tol:
                break
            else:
                self.ll_.append(ll_val)

        # calculate bic
        self.bic = np.log(n_rows)*(sum(self.theta.shape)+len(self.weight)) - 2.0*self.ll_[-1]

    def predict(self, data):
        return np.argmax(self.predict_proba(data), axis=1)

    def predict_proba(self, data):
        return self._calculate_responsibility(data)
