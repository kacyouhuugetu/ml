from scipy.sparse import csr_matrix
from .svm_base import _LOSS_TYPE_SVM_L1_,_LOSS_TYPE_SVM_L2_,_LOSS_TYPE_LOGISTIC_,_get_nl
from ..base import isclose
import numpy as np

def cmls(X,y,w,sample_weights,lambda_,sparse,loss_type,loss_fun,loss_gradient,conv_tol,max_iter,permutation=True,compute_Xy=False,compute_power_abs=False,epsilon=0.1):
	"""
		CMLS/CLG算法。见论文<<Text Categorization Based on Regularized Linear Classification Methods>>
		参数：
			①permutation：bool，表示对变量迭代时是否进行随机迭代。默认为True
			①compute_Xy,compute_power_abs：bool，分别表示是否计算并存储X*y[:,np.newaxis]和是否计算并存储X**2和|X|。计算并存储可以提高算法效率，但需要更多的存储空间
			②epsilon：浮点数，用于更新算法的trust region。默认为0.1
	"""

	N,p = X.shape
	svm_loss = loss_type==_LOSS_TYPE_SVM_L2_
	
	if sparse:
		Xy = csr_matrix(X).multiply(y[:,np.newaxis])
		yz = Xy.dot(w)
		Xy = csr_matrix(Xy.T)
		Xy,Xy_indptr,Xy_indices = Xy.data,Xy.indptr,Xy.indices
		compute_Xy = True
	else:
		yz = y*np.dot(X,w)
		if compute_Xy:
			Xy = np.array(X.T*y,order='C')
	
	if compute_power_abs:
		X_abs,X_power = np.abs(Xy),Xy**2 if compute_Xy else (np.array(np.abs(X).T,order='C'),np.array((X**2).T,order='C'))
	
	if not sparse:
		is_allclose_zero = np.all(np.isclose(X,0),axis=0)
	
	region_size_array = np.ones(p,np.float64)
	sqrt_sample_weights = None if sample_weights is None else np.sqrt(sample_weights)
	variable_indices = np.arange(p,dtype=np.uint32)

	for iter_index in range(max_iter):
		ck = 1-iter_index/(max_iter-1)
		variable_indices = np.random.choice(variable_indices,replace=False,size=p) if permutation else variable_indices
		
		for variable_index in variable_indices:
			region_size = region_size_array[variable_index]
			
			#若数据的某一个变量的所有值都为0，则跳过该变量不进行更新
			if sparse:
				start_index,end_index = Xy_indptr[variable_index],Xy_indptr[variable_index+1]
				if start_index==end_index:
					continue
				xy = Xy[start_index:end_index]
				xy_nz_indices = Xy_indices[start_index:end_index]
			else:
				if is_allclose_zero[variable_index]:
					continue
				xy = Xy[variable_index] if compute_Xy else X[:,variable_index]*y

			if compute_power_abs:
				x_abs,x_power = (X_abs[start_index:end_index],X_power[start_index:end_index]) if sparse else (X_abs[variable_index],X_power[variable_index])
			else:
				x_abs,x_power = np.abs(xy),xy**2
			
			nz_sample_weights = sample_weights[xy_nz_indices] if sparse and not sample_weights is None else sample_weights
			nz_sqrt_sample_weights = sqrt_sample_weights[xy_nz_indices] if sparse and not sample_weights is None else sqrt_sample_weights
			yz_ = yz[xy_nz_indices] if sparse else yz
			
			#计算C和F
			#令fc表示进过"smooth"后的目标函数，则C为fc对r(yz,即y*w'x)的梯度，而F为fc对r的二阶梯度的上界
			#见论文3.1节
			if svm_loss:
				F = np.empty(xy_nz_indices.shape[0] if sparse else N,np.float64)
				Cr = 2*(yz_-1) if nz_sample_weights is None else 2*nz_sample_weights*(yz_-1)
				F[:] = 2 if nz_sample_weights is None else 2*nz_sample_weights
				Cr[yz_>=1] *= ck
				F[yz_>=1+region_size*x_abs] *= ck

			else:
				exp_yz = np.exp(yz_) 
				Cr = -np.reciprocal(1+exp_yz)
				F = np.exp(region_size*x_abs)/(2+exp_yz+np.reciprocal(exp_yz))
				F[F>0.25] = 0.25

			nz_sample_weights = sample_weights[xy_nz_indices] if sparse and not sample_weights is None else sample_weights
			#计算更新量△v的分子和分母
			num = np.dot(xy,Cr) + lambda_*w[variable_index]
			den = np.dot(x_power,F) + lambda_

			if isclose(den,0):
				continue
			
			delta_v = -num/den
			delta_w =	-region_size if delta_v<-region_size else \
						region_size if delta_v>region_size else \
						delta_v
			delta_yz = delta_w*xy
			
			if sparse:
				yz[xy_nz_indices] += delta_yz
			else:
				yz += delta_yz
			w[variable_index] += delta_w
			region_size_array[variable_index] = 2*abs(delta_w) + epsilon 
	
	_,nl_y,nl_z,nl_sample_weights,_ = _get_nl(w,X,y,sample_weights,svm_loss)
	loss = loss_fun(nl_y,nl_z,nl_sample_weights) + 0.5*lambda_*np.sum(w**2)
	return w,loss

