from .svm_base import _LOSS_TYPE_SVM_L1_,_LOSS_TYPE_SVM_L2_,_LOSS_TYPE_LOGISTIC_,_get_nl
from ..base import isclose,less_equal
import numpy as np

def cdsvm(X,y,w,sample_weights,lambda_,sparse,loss_type,loss_fun,loss_gradient,conv_tol,max_iter,permutation=True,compute_power=False,sigma=0.01,beta=0.9):
	"""
		coordinate descent for svm。具体见论文<<Coordinate Descent Method for Large-scale L2-loss Linear Support Vector Machines>>
		参数：
			①permutation：bool，表示对变量迭代时是否进行随机迭代。默认为True
			②compute_power：bool，表示是否计算并存储X**2。计算并存储X**2可以提高算法效率，但需要更多的存储空间
			③sigma,beta：算法参数。具体见论文
	"""

	N,p = X.shape
	if compute_power:
		X_power = X**2
	
	#计算目标函数下降量的泰勒展开对△w(论文中为z)的二阶导数上界，见式(14)
	if not sample_weights is None:
		H = lambda_ + 2*np.dot(sample_weights,X_power if compute_power else X**2)
	else:
		H = lambda_ + 2*np.sum(X_power if compute_power else X**2,axis=0)

	z = X.dot(w)
	nonzero_loss_indices = y*z<1
	variable_indices = np.arange(p)
	
	for iter_index in range(max_iter):
		norm2_square_derivative = 0
		variable_indices = np.random.choice(variable_indices,replace=False,size=p) if permutation else variable_indices

		for variable_index in variable_indices:
			nl_x,nl_y,nl_z = X[nonzero_loss_indices,variable_index],y[nonzero_loss_indices],z[nonzero_loss_indices]
			nl_sample_weights = sample_weights[nonzero_loss_indices] if not sample_weights is None else 1.
			nl_x_power = X_power[nonzero_loss_indices,variable_index] if compute_power else nl_x**2
			
			#计算目标函数下降量的泰勒展开对△w(论文中为z)的一阶导数和二阶导数，见论文式(9)和式(10)
			derivative = lambda_*w[variable_index] - 2*np.dot(nl_x,nl_sample_weights*(nl_y-nl_z))
			second_derivative = lambda_ + 2*np.sum(nl_sample_weights*nl_x_power)
			if isclose(second_derivative,0.):
				continue
			
			#牛顿步
			d = -derivative/second_derivative
			#最大步长，见论文式(14)
			step_length_max = second_derivative/(H[variable_index]/2+sigma)
			
			#线性搜索步长，见论文式(15)和Algorithm 2
			step_length = 1
			while True:
				if less_equal(step_length,step_length_max):
					break
				step_length *= beta
			
			delta_w = step_length*d
			w[variable_index] += delta_w
			z += X[:,variable_index]*delta_w
			nonzero_loss_indices = y*z<1
			norm2_square_derivative += derivative**2
		
		if norm2_square_derivative<=conv_tol:
			break
	
	loss = loss_fun(y[nonzero_loss_indices],z[nonzero_loss_indices]) + 0.5*lambda_*np.sum(w**2)
	return w,loss

def cddual(X,y,alpha,sample_weights,lambda_,sparse,loss_type,loss_fun,loss_gradient,conv_tol,max_iter,permutation=True,compute_Xy=False,shrink=False):
	"""
		coordinate descent for dual svm。具体见论文<<A Dual Coordinate Descent Method for Large-scale Linear SVM>>
		参数：
			①permutation：bool，表示对变量迭代时是否进行随机迭代。默认为True
			②compute_Xy：bool，表示是否计算并存储X*y[:,np.newaxis]。计算并存储可以提高算法效率，但需要更多的存储空间
			③shrink：bool，表示是否使用shrink来提高算法效率。被shrink的样本将在接下来的迭代中不更新其alpha，具体见论文3.2节
	"""

	N,p = X.shape
	indices = np.arange(N) 

	svm_l2_loss = loss_type==_LOSS_TYPE_SVM_L2_
	w = np.dot(X.T,alpha*y)
	U,D = 1./lambda_,lambda_/2
	if not sample_weights is None:
		if svm_l2_loss:
			D = np.reciprocal(sample_weights)*D
		else:
			U = sample_weights*U
	
	if sparse:
		Xy = csr_matrix(X*y[:,np.newaxis])
		Xy,Xy_indptr,Xy_indices = Xy.data,Xy.indptr,Xy.indices
		compute_Xy = True
	elif compute_Xy:
		Xy = X*y[:,np.newaxis]
	
	Q_diag = np.sum(X**2,axis=1) + (D if svm_l2_loss else 0)
	
	sample_indices = indices
	M_,m_ = np.inf,-np.inf
	for iter_index in range(max_iter):
		if shrink:
			new_sample_indices = []
			max_projected_sub_gradient,min_projected_sub_gradient = -np.inf,np.inf
		else:
			norm2_square_gradient = 0
		
		sample_indices = np.random.choice(sample_indices,replace=False,size=len(sample_indices)) if permutation else sample_indices
		for sample_index in sample_indices:
			if svm_l2_loss:
				Dii = D[sample_index] if not sample_weights is None else D
			else:
				upper = U[sample_index] if not sample_weights is None else U

			if sparse:
				start_index,end_index = Xy_indptr[sample_index],Xy_indptr[sample_index+1]
				xy = Xy[start_index:end_index]
				xy_nz_indices = Xy_indices[start_index:end_index]
			else:	
				xy = Xy[sample_index] if compute_Xy else X[sample_index]*y[sample_index]

			alpha_ = alpha[sample_index]
			xy_dot_w = np.dot(xy,w[xy_nz_indices]) if sparse else np.dot(xy,w)
			#计算梯度G，见论文式(12)
			sub_gradient = xy_dot_w - 1 + (alpha_*Dii if svm_l2_loss else 0)
			
			#计算projected梯度，见论文式(8)
			if isclose(alpha_,0):
				projected_sub_gradient = sub_gradient if sub_gradient<0 else 0
				if shrink and less_equal(sub_gradient,M_):
					new_sample_indices.append(sample_index)

			elif not svm_l2_loss and isclose(alpha_,upper):
				projected_sub_gradient = sub_gradient if sub_gradient>0 else 0
				if shrink and less_equal(m_,sub_gradient):
					new_sample_indices.append(sample_index)

			else:
				projected_sub_gradient = sub_gradient
			
			#计算最大和最小projected梯度，用于计算shrink中的M和m，见式(16)
			if shrink:
				if projected_sub_gradient>max_projected_sub_gradient:
					max_projected_sub_gradient = projected_sub_gradient
				if projected_sub_gradient<min_projected_sub_gradient:
					min_projected_sub_gradient = projected_sub_gradient
			else:
				norm2_square_gradient += projected_sub_gradient**2

			#更新alpha，见式(9)
			if not isclose(projected_sub_gradient,0):
				new_alpha = alpha_ - sub_gradient/Q_diag[sample_index]
				alpha[sample_index] = 0 if new_alpha<0 else \
										upper if not svm_l2_loss and new_alpha>upper else \
										new_alpha
				if sparse:
					w[xy_nz_indices] += (alpha[sample_index] - alpha_)*xy	
				else:
					w += (alpha[sample_index] - alpha_)*xy
		
		#检查算法是否收敛
		#见论文Algorithm 3 while循环第3步
		if shrink:
			if max_projected_sub_gradient-min_projected_sub_gradient<=conv_tol:
				if len(sample_indices) == N:
					break
				else:
					sample_indices = indices
					M_,m_ = np.inf,-np.inf
			else:
				sample_indices = new_sample_indices if len(new_sample_indices)>0 else indices
				M_ = max_projected_sub_gradient if max_projected_sub_gradient>0 else np.inf
				m_ = min_projected_sub_gradient if min_projected_sub_gradient<0 else -np.inf

		elif norm2_square_gradient<=conv_tol:
			break
	
	_,nl_y,nl_z,nl_sample_weights,_ = _get_nl(w,X,y,sample_weights,True)
	loss = loss_fun(nl_y,nl_z,nl_sample_weights) + 0.5*lambda_*np.sum(w**2)
	return w,loss

