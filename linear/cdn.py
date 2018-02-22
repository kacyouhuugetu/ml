from ..base import isclose,less_equal
from .l1_base import get_x_nz_indices
import numpy as np

def get_x_nz_indices(X,variable_index,sparse,is_allclose_zero,slice_all):
	if sparse:
		start_index,end_index = X[1][variable_index],X[1][variable_index+1]
		if start_index==end_index:
			return None,None
		x = X[0][start_index:end_index]
		x_nz_indices = X[2][start_index:end_index]
	else:
		if is_allclose_zero[variable_index]:
			return None,None
		x = X[:,variable_index]
		x_nz_indices = slice_all
	return x,x_nz_indices

def CDN(X,y,w,sample_weights,C,sparse,conv_tol,max_iter,permutation=True,sigma=0.01,beta=0.5,stop_criterion=None,stop_criterion_ratio=0.01):
	"""
		Coordinate Descent method using one-dimensional Newton direction(CDN)。具体见论文<<A Comparison of Optimization Methods and Software for Large-scale L1-regularized Linear Classification>> 4.1.2节，以及论文<<LIBLINEAR A Library for Large Linear Classification>> Appendix G
		参数：
			①permutation：bool，表示对变量迭代时是否进行随机迭代。默认为True
			②stop_criterion：浮点数，表示收敛精度。默认为None
			③stop_criterion_ratio：浮点数。若stop_criterion为None，将自动指定stop_criterion为stop_criterion_ratio*min_class_samples_ratio*violation，其中min_class_samples_ratio = min(#pos,#neg)/N，violation是第一次迭代时算法的violation
			④sigma,beta：浮点数，用于线性搜索中
	"""
	N,p = X.shape
	
	if sparse:
		X = csr_matrix(X.T)
		X_data,X_indptr,X_indices = X.data,X.indptr,X.indices
		X = csr_matrix(X.T)
		X_ = (X_data,X_indptr,X_indices)
		slice_all,is_allclose_zero = None,None
	else:
		X_ = X
		slice_all,is_allclose_zero = slice(N),np.all(np.isclose(X,0),axis=0)
	
	exp_neg_Xy_dot_w = np.exp(-y*X.dot(w))
	loss = np.sum(np.log(1+exp_neg_Xy_dot_w) * (1 if sample_weights is None else sample_weights))
	indices = np.arange(p,dtype=np.uint32)

	#keep_indices表示没有被shrinking的变量索引
	keep_indices = np.ones(p,np.bool8)
	variable_indices = indices
	M = np.inf

	for iter_index in range(max_iter):
		violation,max_abs_subgradient = 0.,-np.inf

		variable_indices = np.random.choice(variable_indices,replace=False,size=variable_indices.shape[0]) if permutation else variable_indices
		for variable_index in variable_indices:
			x,x_nz_indices = get_x_nz_indices(X_,variable_index,sparse,is_allclose_zero,slice_all)
			if x is None:
				w[variable_index] = 0.
				keep_indices[variable_index] = False
				continue

			xy,w_,sample_weights_ = x*y[x_nz_indices],w[variable_index],1 if sample_weights is None else sample_weights[x_nz_indices]
			nz_exp_neg_Xy_dot_w = exp_neg_Xy_dot_w[x_nz_indices]
			#计算sample_weights*{ σ(yw'x)*[1-σ(yw'x)] }
			D = np.reciprocal(nz_exp_neg_Xy_dot_w+np.reciprocal(nz_exp_neg_Xy_dot_w)+2.)*sample_weights_
			#计算sample_weights*[σ(yw'x)-1]
			probs_minus_one = (np.reciprocal(1+nz_exp_neg_Xy_dot_w)-1)*sample_weights_
			gradient = C * np.dot(probs_minus_one,xy)
			hessian = C * np.dot(x,D*x)
			
			#进行shrinking，见论文<<A Comparison of Optimization Methods and Software for Large-scale L1-regularized Linear Classification>> 4.1.2节 式(31)、式(32)
			if isclose(w_,0) and abs(gradient)<1-M/N:
				keep_indices[variable_index] = False
			else:
				d = -(gradient+1)/hessian if less_equal(gradient+1,hessian*w_) else \
					-(gradient-1)/hessian if less_equal(hessian*w_,gradient-1) else \
					-w_
				abs_subgradient = abs( gradient+1 if w_>0 else gradient-1 if w_<0 else abs(gradient)-1 )
				max_abs_subgradient = max(max_abs_subgradient,abs_subgradient)
				violation += abs_subgradient

				_lambda = 1
				abs_w,abs_w_plus_d = abs(w_),abs(w_+d)

				#在论文<<LIBLINEAR A Library for Large Linear Classification>>中，作者推荐在line search中使用目标函数decrease的一个上界，这个上界易于计算，可以避免多余的log计算。
				#缺点是这个上界要求对当前的variable index j，对所有样本有xij>0。当数据稀疏时，这一条件无法实现
				#目前本算法并没有实现这个上界，而是简单进行穷举
				#进行线性搜索，见论文 4.1.2节式(28)
				while True:
					nz_exp_neg_Xy_dot_w_ = nz_exp_neg_Xy_dot_w/np.exp(_lambda*d*xy)
					abs_new_w = abs(w_+_lambda*d)
					loss_ = np.sum(sample_weights_*np.log(1+nz_exp_neg_Xy_dot_w_))
					
					condition = gradient*d + abs_w_plus_d - abs_w
					if less_equal(loss_-loss+abs_new_w-abs_w,sigma*_lambda*condition):
						loss = loss_
						exp_neg_Xy_dot_w[x_nz_indices] = nz_exp_neg_Xy_dot_w_
						break
					_lambda *= beta

				w[variable_index] += _lambda*d

		#若算法不指定stop_criterion，我们根据算法第一次迭代的violation来设定stop criterion
		#见论文<<LIBLINEAR: A Library for Large Linear Classification>> Appendix F
		if iter_index==0 and stop_criterion is None:
			min_class_samples_ratio = np.unique(y,return_counts=True)[1].min()
			stop_criterion = stop_criterion_ratio*min_class_samples_ratio*violation
		
		#在本算法中，我们选取violation的1-范数来作为是否收敛的根据
		#其他可用的收敛根据有violation的2-范数或∞-范数等
		if less_equal(violation,stop_criterion):
			if np.all(keep_indices):
					break
			else:
				keep_indices = np.ones(p,np.bool8)
				M = np.inf
		else:
			M = max_abs_subgradient
			variable_indices = indices[keep_indices]

	return w

