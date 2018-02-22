import numpy as np

def SGD(X,y,w,sample_weights,C,sparse,conv_tol,max_iter,permutation=True):
	N,p = X.shape
	C = C if sample_weights is None else C*sample_weights[:,np.newaxis]
	hessian = 0.25*np.sum(C*X**2,axis=0)
	indices = np.arange(p*2,dtype=np.uint32)
	C = C if sample_weights is None else C.ravel()

	w_ = np.zeros(p*2,np.float64)
	if not w is None:
		pos_indices,neg_indices = w>0,w<0 
		w_[:p][pos_indices] = w[pos_indices]
		w_[p:][neg_indices] = w[neg_indices]
	
	exp_neg_Xy_dot_w = np.exp(-y*np.dot(X,w))
	for iter_index in range(max_iter):
		variable_indices = np.random.choice(indices,replace=False,size=p*2) if permutation else indices
		for variable_index_ in variable_indices:
			variable_index,sign = (variable_index_,1) if variable_index_<p else \
									(variable_index_-p,-1)
			xy = X[:,variable_index]*y
			
			probs_minus_one = C * ( np.reciprocal(1+exp_neg_Xy_dot_w)-1 )
			gradient = sign*np.dot(probs_minus_one,xy) + 1
			delta_w = max(-w_[variable_index_],-gradient/hessian[variable_index])
			w_[variable_index_] += delta_w

			exp_neg_Xy_dot_w /= np.exp(sign*delta_w*xy)

	return w_[:p]-w_[p:]
