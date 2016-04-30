def _init_variables(X,K):
    N,D = np.shape(X)
    pi_1 = sc.stats.dirichlet.rvs(np.ones(shape=K))[0]
    mu_1 = sc.stats.multivariate_normal.rvs(mean=np.mean(X,axis=0),cov=np.eye(N=D,M=D),size=K)
    cov_1 = np.zeros((K,D,D))
    for k in range(K): 
        cov_1[k,:,:] = sc.stats.uniform.rvs()*np.eye(N=D,M=D)
    return pi_1,mu_1,cov_1

def _responsibility(X,pi,mu,cov):
    N,D = np.shape(X)
    K = len(pi)
    r_numerator = np.zeros(shape=(N,K))
    for k in range(K):
        r_numerator[:,k] = pi[k]*sc.stats.multivariate_normal.pdf(X,mean=mu[k],cov=cov[k])
    r_denominator = np.sum(r_numerator,axis=1)
    r = r_numerator/np.tile(r_denominator,(K,1)).T
    return r
    
def em_gmm(X,K,max_iter,tol):
    N,D = np.shape(X)
    ll = [-np.inf]
    pi_i,mu_i,cov_i = _init_variables(X,K)
    iterations = range(max_iter)
    for i in iterations:
        # E-step. Calculate responsibilities
        r = _responsibility(X,pi_i,mu_i,cov_i)

        # M-step. Maximize parameters
        # pi
        for k in range(K):
            pi_i[k] = np.sum(r[:,k])/float(N)

        # mu
        for k in range(K):
            mu_numerator = np.zeros((N,D))
            for n in range(N):
                mu_numerator[n,:] = r[n,k]*X[n,:]
            mu_numerator = np.sum(mu_numerator,axis=0)
            mu_denominator = np.sum(r[:,k])
            mu_i[k] = mu_numerator/mu_denominator

        # cov
        for k in range(K):
            cov_numerator = np.zeros((N,D,D))
            for n in range(N):
                aux = X[n,:]-mu_i[k]
                cov_numerator[n,:,:] = r[n,k]*np.dot(aux[:,None],aux[:,None].T)
            cov_numerator = np.sum(cov_numerator,axis=0)
            cov_denominator = np.sum(r[:,k])
            cov_i[k] = cov_numerator/cov_denominator

        # Calculate log-likelihood
        aux = np.zeros(shape=(N,K))
        for k in range(K):
            normal_prob = sc.stats.multivariate_normal.pdf(X,mean=mu_i[k],cov=cov_i[k])
            aux[:,k] = pi_i[k]*normal_prob
        ll_val = np.sum(np.log(np.sum(aux,axis=1)))
        if np.abs(ll_val-ll[-1])<tol:
            break
        else:
            ll.append(ll_val)
            
    return {
        'pi':pi_i,
        'mu':mu_i,
        'cov':cov_i,
        'll':ll
    }