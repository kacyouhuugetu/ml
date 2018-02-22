import numpy as np

def consistent_estimator(X,sample_cov=None,return_shrinkage_coef=False,return_shrinkage_cov=False):
	N,p = X.shape
	if return_shrinkage_cov:
		return_shrinkage_coef = True

	if sample_cov is None or shrinkage_coef:
		mean = np.mean(X,axis=0)
		diff = X-mean
		sample_cov = np.tensordot(diff,diff,axes=(0,0))/N
	
	std = np.sqrt(np.diag(sample_cov))
	shrinkage_target = np.outer(std,std)
	corr = sample_cov/shrinkage_target

	ave_corr = np.mean(corr[np.triu_indices(p,k=1)])
	shrinkage_target *= ave_corr
	shrinkage_target[np.diag_indices(p)] = np.diag(sample_cov)
	
	if return_shrinkage_coef:
		diff = np.einsum('ij,ik->jki',diff,diff)-sample_cov[:,:,np.newaxis]

		asy_cov = np.tensordot(diff,diff,axes=(2,2))/N
		std_ratio = np.outer(np.reciprocal(std),std)
		
		pi = np.einsum('ijij->',asy_cov)
		pi_ii = np.einsum('iiii->',asy_cov)
		rho = ( np.einsum('iiij,ij->',asy_cov,std_ratio) + np.einsum('iiji,ij->',asy_cov,std_ratio) - 2*pi_ii )*ave_corr/2
		rho += pi_ii
		gamma = np.sum((sample_cov-shrinkage_target)**2)

		k = (pi-rho)/gamma
		shrinkage_coef = max(0,min(k/N,1))
		
		if return_shrinkage_cov:
			return shrinkage_coef*shrinkage_target + (1-shrinkage_coef)*sample_cov
		
		return shrinkage_target,shrinkage_coef
	
	return shrinkage_target

def OAS(X,sample_cov=None,return_shrinkage_coef=False,return_shrinkage_cov=False,niter=100,conv_tol=1e-6):
	N,p = X.shape
	if return_shrinkage_cov:
		return_shrinkage_coef = True
	
	if sample_cov is None:
		sample_cov = np.cov(X,rowvar=False)
	shrinkage_target = np.trace(sample_cov)/p
	
	if return_shrinkage_coef:
		trace_square_sample_cov = np.trace(np.dot(sample_cov,sample_cov))
		trace_square_shrinkage_target = shrinkage_target**2
		shrinkage_coef = ( (1-2/p) * trace_square_sample_cov + trace_square_shrinkage_target/p**2 ) / ( (N+1-2/p) * trace_square_sample_cov - (1-N/p) * trace_square_shrinkage_target/p )
		
		shrinkage_coef = max(0,min(shrinkage_coef,1))
		
		if return_shrinkage_cov:
			return shrinkage_coef*shrinkage_target + (1-shrinkage_coef)*sample_cov
		
		return shrinkage_target,shrinkage_coef

	return shrinkage_target

