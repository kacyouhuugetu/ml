from ..base import isclose,logistic_fun
import numpy as np

def BBR(X,y,w,sample_weights,C,sparse,conv_tol,max_iter,permutation=True,compute_Xy=False,compute_power_abs=False):
	"""
		Bayesian Binary Regression(BBR)算法。具体见论文<<Large-Scale Bayesian Logistic Regression for Text Categorization>>
		参数：
			①permutation：bool，表示对变量迭代时是否进行随机迭代。默认为True
			②compute_Xy,compute_power_abs：bool，分别表示是否计算并存储X*y[:,np.newaxis]和是否计算并存储X**2和|X|。计算并存储可以提高算法效率，但需要更多的存储空间
	"""

	N,p = X.shape
	variable_indices = np.arange(p,dtype=np.uint32)

	if sparse:
		Xy = csr_matrix(X).multiply(y[:,np.newaxis])
		r = Xy.dot(w)
		Xy = csr_matrix(Xy.T)

		Xy_data,Xy_indptr,Xy_indices = Xy.data,Xy.indptr,Xy.indices
		Xy = (Xy_data,Xy_indptr,Xy_indices)
		compute_Xy = True
		slice_all,is_allclose_zero = None,None
	else:
		r = y*np.dot(X,w)
		slice_all,is_allclose_zero = slice(N),np.all(np.isclose(X,0),axis=0)
		if compute_Xy:
			Xy = np.array(X.T*y,order='C')
	
	if compute_power_abs:
		X_abs,X_power = np.abs(Xy),Xy**2 if compute_Xy else (np.array(np.abs(X).T,order='C'),np.array((X**2).T,order='C'))
	
	F = np.empty(N,np.float64)
	region_size_array = np.ones(p,np.float64)
	for iter_index in range(max_iter):

		variable_indices = np.random.choice(variable_indices,replace=False,size=p) if permutation else variable_indices
		
		for variable_index in variable_indices:
			region_size = region_size_array[variable_index]

			#若数据的某一个变量的所有值都为0，则跳过该变量不进行更新
			if compute_Xy:
				xy,xy_nz_indices = get_x_nz_indices(Xy,variable_index,sparse,is_allclose_zero,slice_all)
				if xy is None:
					continue
			else:
				if is_allclose_zero[variable_index]:
					continue
				xy,xy_nz_indices = y*X[:,variable_index],slice_all
			
			if compute_power_abs:
				x_abs,x_power = (X_abs[start_index:end_index],X_power[start_index:end_index]) if sparse else (X_abs[variable_index],X_power[variable_index])
			else:
				x_abs,x_power = np.abs(xy),xy**2
			
			sample_weights_ = 1 if sample_weights is None else sample_weights[xy_nz_indices]
			r_,w_ = r[xy_nz_indices] if sparse else r,w[variable_index]
			diff = np.abs(r_)-region_size*x_abs
			
			larger_indices = diff>0
			exp_diff = np.exp(diff[larger_indices])

			#计算F值，见论文4.2节
			if sparse:
				F[xy_nz_indices] = 0.25
				F[xy_nz_indices[larger_indices]] = np.reciprocal( 2. + exp_diff + np.reciprocal(exp_diff) ) 
			else:
				F[:] = 0.25
				F[larger_indices] = np.reciprocal( 2. + exp_diff + np.reciprocal(exp_diff) ) 
			
			#见论文式(12)和式(13)
			neg_gradient = C * np.dot(sample_weights_*logistic_fun(-r_),xy)
			hessian = C * np.dot(F[xy_nz_indices]*sample_weights_,x_power)
			
			#见论文式(18)
			delta_v_pos,delta_v_neg = (neg_gradient-1)/hessian,(neg_gradient+1)/hessian
			
			#若w为0，neg_gradient在此处无定义
			#为了确保目标函数下降，我们选择delta_v使得目标函数下降
			#具体可见论文<<A Comparison of Optimization Methods and Software for Large-scale L1-regularized Linear Classification>>4.1.1节
			if isclose(w_,0):
				delta_v = delta_v_pos if delta_v_pos>0 else \
						delta_v_neg if delta_v_neg<0 else \
						0
			#若w不为0，则存在neg_gradient
			#利用式(20)计算△ v
			else:
				delta_v,is_pos = (delta_v_pos,True) if w_>0 else (delta_v_neg,False)
				new_w = w_ + delta_v
				if ( new_w<0 and is_pos ) or ( new_w>0 and not is_pos ):
					delta_v = -w_
			
			#△w计算，确保△w在trust region [-region_size,region_size]内
			#见式(19)
			delta_w = -region_size if delta_v<-region_size else region_size if delta_v>region_size else delta_v
			r[xy_nz_indices] += delta_w*xy
			w[variable_index] += delta_w
			#region size更新，见式(16)
			region_size_array[variable_index] = max(2*abs(delta_w),region_size/2)

	return w
