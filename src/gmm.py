
import numpy as np
import scipy as sc

class GMM():

    def __init__(self, n_components=2, random_state=None, tol=1e-3, n_iter=100):
        self.n_components = n_components
        self.random_state = random_state
        self.tol = tol
        self.n_iter = n_iter

        # flag to indicate if converged
        self.converged_ = False

    def _do_e_step(self, X):

        n_rows, n_cols = np.shape(X)
        r_numerator = np.zeros(shape=(n_rows,self.n_components))
        for k in range(self.n_components):
            r_numerator[:,k] =
            	self.weigths_[k]*sc.stats.multivariate_normal.pdf(X,
            		mean=self.means_[k],cov=self.covars_[k])
        r_denominator = np.sum(r_numerator,axis=1)
        r = r_numerator/np.tile(r_denominator,(self.n_components,1)).T
        return r

    def _do_m_step(self, X, responsibility):

        n_rows, n_cols = np.shape(X)

        # pi
        for k in range(self.n_components):
            self.weigths_[k] = np.sum(responsibility[:,k])/float(n_rows)

        # mu
        for k in range(self.n_components):
            mu_numerator = np.zeros((n_rows,n_cols))
            for n in range(n_rows):
                mu_numerator[n,:] = responsibility[n,k]*X[n,:]
            mu_numerator = np.sum(mu_numerator,axis=0)
            mu_denominator = np.sum(responsibility[:,k])
            self.means_[k] = mu_numerator/mu_denominator

        # cov
        for k in range(self.n_components):
            cov_numerator = np.zeros((n_rows,n_cols,n_cols))
            for n in range(n_rows):
                aux = X[n,:]-self.means_[k]
                cov_numerator[n,:,:] = responsibility[n,k]*
                    np.dot(aux[:,None],aux[:,None].T)
            cov_numerator = np.sum(cov_numerator,axis=0)
            cov_denominator = np.sum(responsibility[:,k])
            self.covars_[k] = cov_numerator/cov_denominator


    def fit(self, X, y=None):

        # initialization step
        n_rows, n_cols = np.shape(X)
        if n_rows < self.n_components:
            raise ValueError(
                '''
                GMM estimation with {n_components} components, but got only
                {n_rows} samples
                '''.format(self.n_components,n_rows))

        if self.verbose > 0:
            print('EM algorithm started')

        self.weigths_ = sc.stats.dirichlet.rvs(np.ones(shape=n_components))[0]

        self.means_ = sc.stats.multivariate_normal.rvs(
                    mean=np.mean(X,axis=0),
                    cov=np.eye(N=n_cols,M=n_cols),
                    size=self.n_components)

        covars = np.zeros((self.n_components,n_cols,n_cols))
        for k in self.n_components:
            covars[k,:,:] = sc.stats.uniform.rvs()*np.eye(N=n_cols,M=n_cols)
        self.covars_ = covars

        self.ll_ = []

        for i in range(self.n_iter):
            if self.verbose > 0:
                print('\tEM iteration {n_iter}'.format(i))

            # E-step
            responsibilities = _do_e_step(X)

            # M-step
            _do_m_step(X, responsibilities)

            # Check for convergence
            aux = np.zeros(shape=(N,K))
	        for k in range(K):
	            normal_prob = sc.stats.multivariate_normal.pdf(X,
	            	mean=self.means_[k],cov=self.covars_[k])
	            aux[:,k] = self.weigths_[k]*normal_prob
	        ll_val = np.sum(np.log(np.sum(aux,axis=1)))
	        if np.abs(ll_val-ll[-1])<self.tol:
	            break
	        else:
	            self.ll_.append(ll_val)
	            