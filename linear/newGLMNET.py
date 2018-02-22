from numpy.linalg import norm
from ..base import isclose,less_equal
from .l1_base import get_x_nz_indices
import numpy as np

def newGLMNET(X,y,w,sample_weights,C,sparse,conv_tol,max_iter,permutation=True,sigma=0.01,beta=0.5,stop_criterion_out=1e-3,stop_criterion_in=None,stop_criterion_out_ratio=0.01,max_inner_iter=100,shrink_stop_criterion_in_max_iter=1,shrink_ratio=0.25,min_stop_criterion_in=1e-6,zero_atol=1e-30):
	"""
		newGLMNET算法。详细见论文<<An Improved GLMNET for L1-regularized Logistic Regression>>
		参数：
			①permutation：bool，表示对变量迭代时是否进行随机迭代。默认为True
			②stop_criterion_out,stop_criterion_in：浮点数，表示收敛精度。
			③stop_criterion_out_ratio：浮点数。若stop_criterion_out为None，将自动指定stop_criterion_out为stop_criterion_out_ratio*min_class_samples_ratio*violation，其中min_class_samples_ratio = min(#pos,#neg)/N，violation是第一次外部迭代时算法的violation
			④max_inner_iter：浮点数，表示内部最大迭代次数。默认为100
			⑤shrink_stop_criterion_in_max_iter,shrink_ratio,min_stop_criterion_in：用于缩小内部收敛精度，见论文式(28)
			⑥sigma,beta：浮点数，用于线性搜索中
	"""
	N,p = X.shape
	if w is None:
		w = np.zeros(p,np.float64)

	if sparse:
		X = csr_matrix(X.T)
		X_data,X_indptr,X_indices = X.data,X.indptr,X.indices
		X = csr_matrix(X.T)
		X_ = (X_data,X_indptr,X_indices)
		slice_all,is_allclose_zero = None,None
	else:
		X_ = X
		slice_all = slice(N)
		is_allclose_zero = np.all(np.isclose(X,0,atol=zero_atol),axis=0)
	
	C = C if sample_weights is None else C*sample_weights

	d = np.empty(p,np.float64)
	X_dot_d = np.empty(N,np.float64)
	exp_neg_Xy_dot_w = np.exp(-y*X.dot(w))
	norm1_w = norm(w,ord=1)
	loss = np.sum(C*np.log(1+exp_neg_Xy_dot_w)) + norm1_w
	indices = np.arange(p,dtype=np.uint32)

	M_out = np.inf
	outer_gradient = np.empty(p,np.float64)
	outer_hessian = np.empty(p,np.float64)
	for outer_iter_index in range(max_iter):
		
		#计算C*{ σ(yw'x)*[1-σ(yw'x)] }
		D = np.reciprocal(exp_neg_Xy_dot_w+np.reciprocal(exp_neg_Xy_dot_w)+2.)*C
		#计算C*[σ(yw'x)-1]
		probs_minus_one = (np.reciprocal(1+exp_neg_Xy_dot_w)-1)*C
		
		#keep_indices表示没有被外部shrinking的变量索引
		keep_indices = np.ones(p,np.bool8)
		violation,max_abs_subgradient = 0.,-np.inf
		
		#获取二次模型并进行外部shrinking
		for variable_index in range(p):
			x,x_nz_indices = get_x_nz_indices(X_,variable_index,sparse,is_allclose_zero,slice_all)
			if x is None:
				w[variable_index] = 0
				keep_indices[variable_index] = False
				continue
			
			w_ = w[variable_index]
			outer_gradient[variable_index] = gradient = np.dot(probs_minus_one[x_nz_indices],x*y[x_nz_indices])
			#也可以用预先计算x**2来计算outer hessian
			outer_hessian[variable_index] = np.dot(x,D[x_nz_indices]*x)
			#进行外部shrinking
			if isclose(w_,0) and abs(gradient) < 1-M_out/N:
				keep_indices[variable_index] = False
			else:
				abs_subgradient = abs( gradient+1 if w_>0 else gradient-1 if w_<0 else abs(gradient)-1 )
				violation += abs_subgradient
				max_abs_subgradient = max(max_abs_subgradient,abs_subgradient)
		
		if outer_iter_index==0:
			#若不指定外部迭代的收敛精度，则指定其为stop_criterion_out_ratio*min_class_samples_ratio*violation，其中min_class_samples_ratio = min(#pos,#neg)/N
			if stop_criterion_out is None:
				min_class_samples_ratio = np.unique(y,return_counts=True)[1].min()
				stop_criterion_out = stop_criterion_out_ratio*min_class_samples_ratio*violation
			#若不指定内部迭代的收敛精度，则指定其为第一次外部迭代时的violation
			if stop_criterion_in is None:
				stop_criterion_in = violation
				
		if less_equal(violation,stop_criterion_out):
			break

		M_out = max_abs_subgradient
		#variable_indices表示没有被外部shrinking的变量索引
		variable_indices = indices[keep_indices]
		n_free_variables = variable_indices.shape[0]

		d[:] = 0.
		X_dot_d[:] = 0.

		M_in = np.inf
		#variable_indices[keep_indices]表示没有被内部shrinking的变量索引
		keep_indices = np.ones(n_free_variables,np.bool8)
		#使用coordinate descent方法求解最优步d，并进行内部shrinking
		for inner_iter_index in range(max_inner_iter):
			
			violation,max_abs_subgradient = 0.,-np.inf
			inner_indices = np.random.choice(range(n_free_variables),replace=False,size=n_free_variables) if permutation else range(n_free_variables)
			for index in inner_indices:
				if not keep_indices[index]:
					continue

				variable_index = variable_indices[index]
				x,x_nz_indices = get_x_nz_indices(X_,variable_index,sparse,is_allclose_zero,slice_all)
				new_w = w[variable_index]+d[variable_index]

				inner_gradient = np.dot(x,D[x_nz_indices]*X_dot_d[x_nz_indices]) + outer_gradient[variable_index]
				inner_hessian = outer_hessian[variable_index]
				#进行内部shrinking
				if isclose(new_w,0) and abs(inner_gradient)<1-M_in/N:
					keep_indices[index] = False
				else:
					z = -(inner_gradient+1)/inner_hessian if less_equal(inner_gradient+1,inner_hessian*new_w) else \
						-(inner_gradient-1)/inner_hessian if less_equal(inner_hessian*new_w,inner_gradient-1) else \
						-new_w
					d[variable_index] += z
					X_dot_d[x_nz_indices] += z*x
					
					abs_subgradient = abs( inner_gradient+1 if new_w>0 else inner_gradient-1 if new_w<0 else abs(inner_gradient)-1 )
					max_abs_subgradient = max(max_abs_subgradient,abs_subgradient)
					violation += abs_subgradient

			if less_equal(violation,stop_criterion_in):
				if np.all(keep_indices):
					break
				else:
					keep_indices = np.ones(n_free_variables,np.bool8)
					M_in = np.inf
			else:
				M_in = max_abs_subgradient
		
		#若在shrink_stop_criterion_in_max_iter个迭代内，内部迭代就已收敛，则我们按比例缩小stop_criterion_in
		#若stop_criterion_in已缩小至min_stop_criterion_in，则不再进行缩小
		if inner_iter_index<shrink_stop_criterion_in_max_iter and stop_criterion_in>min_stop_criterion_in:
			stop_criterion_in *= shrink_ratio
		
		lambda_ = 1
		norm1_w_plus_d = norm(w+d,ord=1)
		#进行线性搜索，确定步长
		while True:
			#注意：当X_dot_d某一元素为绝对值很大的负数时，会造成结果上溢出
			#当检测到这一情况，简单地提前退出算法
			if X_dot_d.min()<-200:
				return w

			exp_neg_Xy_dot_w_ = exp_neg_Xy_dot_w/np.exp(lambda_*y*X_dot_d)
			norm1_new_w = norm(w+lambda_*d,ord=1)
			new_loss = np.sum(C*np.log(1+exp_neg_Xy_dot_w_)) + norm1_new_w
			
			condition = np.dot(outer_gradient,d) + norm1_w_plus_d - norm1_w
			if less_equal(new_loss-loss,sigma*lambda_*condition):
				loss,norm1_w = new_loss,norm1_new_w
				exp_neg_Xy_dot_w = exp_neg_Xy_dot_w_
				break

			lambda_ *= beta

		w += lambda_*d

	return w
