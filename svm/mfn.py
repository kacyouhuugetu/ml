from numpy.linalg import norm
from scipy.sparse import csr_matrix,isspmatrix_csr
from ..base import isclose,less_equal
import numpy as np

#用于求解l2正则线性回归的CG算法
#见论文Step 3 Algorithm CGLS
def cgls(w,lambda_,X,X_T,y,z,sample_weights,epsilon,norm_X,max_iter):
	w = w.copy()
	z = (y-z) if sample_weights is None else sample_weights*(y-z)
	r = X_T.dot(z if sample_weights is None else sample_weights*z)-lambda_*w
	p = r.copy()
	norm2_square_r = np.sum(r**2)

	optimality = False
	for iter_index in range(max_iter):
		q = X.dot(p) if sample_weights is None else sample_weights*X.dot(p)
		gamma = norm2_square_r/(np.sum(q**2)+lambda_*np.sum(p**2))
		w += gamma*p
		z -= gamma*q
		
		r = X_T.dot(z if sample_weights is None else sample_weights*z)-lambda_*w
		norm2_square_r_ = np.sum(r**2)
		if less_equal(norm2_square_r_,(epsilon*(norm_X*norm(z)+lambda_*norm(w)))**2):
			optimality = True
			break

		omega = norm2_square_r_/norm2_square_r
		norm2_square_r = norm2_square_r_
		p *= omega
		p += r

	return w,optimality

#进行线性搜索求解最优步长
#见论文Step 4
def line_search(X,y,sample_weights,lambda_,w_new,w,z_new,z,nonzero_loss_indices,loss_fun,loss):
	z_diff = z_new-z
	y_prod_z_diff = y*z_diff
	l_prod = z*z_diff-y_prod_z_diff if sample_weights is None else sample_weights*(z*z_diff-y_prod_z_diff)
	r_prod = z_new*z_diff-y_prod_z_diff if sample_weights is None else sample_weights*(z_new*z_diff-y_prod_z_diff)
	l = lambda_*np.dot(w,w_new-w)+np.sum(l_prod[nonzero_loss_indices]) 
	r = lambda_*np.dot(w_new,w_new-w)+np.sum(r_prod[nonzero_loss_indices])
	
	#计算jump point，见论文式(14)和式(15)
	jump_points = (y-z)/z_diff
	jump_points_type1_indices = np.logical_and(nonzero_loss_indices,y_prod_z_diff>0)
	jump_points_type2_indices = np.logical_and(np.logical_not(nonzero_loss_indices),y_prod_z_diff<0)
	jump_points_indices = np.nonzero(np.logical_or(jump_points_type1_indices,jump_points_type2_indices))[0]
	del z_diff,jump_points_type1_indices,jump_points_type2_indices
	
	jump_points_indices = jump_points_indices[np.argsort(jump_points[jump_points_indices])]
	jump_points = jump_points[jump_points_indices]
	y_prod_z_diff = y_prod_z_diff[jump_points_indices]
	l_prod = l_prod[jump_points_indices]
	r_prod = r_prod[jump_points_indices]
	del jump_points_indices
	
	current_jump_point = 0

	min_dist = np.inf
	reach_zero_point,zero_point_ = False,None
	for index in range(len(jump_points)):
		next_jump_point = jump_points[index]
		if not isclose(l,r):
			#求解零点，并确定零点是否在两个jump point之间
			#若是，则该零点为所求步长
			zero_point = l/(l-r)
			if less_equal(current_jump_point,zero_point) and zero_point<next_jump_point:
				reach_zero_point = True
				break
			
			#有时会出现没有位于两个jump point之间的零点
			#则我们找这么一个次零点，其与某一jump point之间的距离最短
			dist_l,dist_r = abs(current_jump_point-zero_point),abs(next_jump_point-zero_point)
			if dist_l<min_dist:
				zero_point_ = current_jump_point
				min_dist = dist_l
			if dist_r<min_dist:
				zero_point_ = next_jump_point
				min_dist = dist_r

		current_jump_point = next_jump_point
		if y_prod_z_diff[index]>0:
			l-=l_prod[index]
			r-=r_prod[index]
		else:
			l+=l_prod[index]
			r+=r_prod[index]
	
	if zero_point_ is None:
		return w,z,nonzero_loss_indices,loss
	if not reach_zero_point:
		zero_point = zero_point_

	#确保零点能够使得目标函数下降
	w_new,z_new = zero_point*w_new + (1-zero_point)*w,zero_point*z_new + (1-zero_point)*z
	nonzero_loss_indices = y*z_new<1
	loss_ = loss_fun(y[nonzero_loss_indices],z_new[nonzero_loss_indices]) + 0.5*lambda_*np.sum(w_new**2)
	if less_equal(loss_,loss):
		return w_new,z_new,nonzero_loss_indices,loss_
	else:
		return w,z,nonzero_loss_indices,loss

def mfn(X,y,w,sample_weights,lambda_,sparse,loss_type,loss_fun,loss_gradient,conv_tol,max_iter,heuristic_1=False,heuristic_2=False,epsilon=1e-6,max_iter_cg=100,conv_tol_heuristic=1e-2,max_iter_cg_heuristic=10):
	"""
		Modified Finite Newton(MFN)算法。具体见论文<<A Modified Finite Newton Method for Fast Solution of Large Scale Linear SVMs>>
		参数：
			①heuristic_1,heuristic_2：bool，表示是否使用heuristic来加速算法。若heuristic_1为True，且初始解向量都为0，则将在算法第一次使用CGLS算法时，使用一个较小的参数max_iter_cg(默认为10)。若heuristic_2为True，则算法会首先使用一个较大的conv_tol(默认为1e-2)来获取解向量，再次执行算法并将上一次算法获得的解向量作为初始解向量。具体见论文4.6节
			②epsilon：浮点数，用于求解牛顿步的CG算法的stop criteria。见式(13)
			③max_iter_cg：int，表示用于求解牛顿步的CG算法的最大迭代次数，默认为100
			④conv_tol_heuristic,max_iter_cg_heuristic：浮点数和int，分别用于heuristic_2与heuristic_1
	"""

	N,p = X.shape
	if isspmatrix_csr(X):
		sparse = True
	elif sparse:
		X = csr_matrix(X)
	
	if np.allclose(w,0.):
		heuristic_1 = False

	if heuristic_2:
		w,_ = _l2_svm_mfn(X,y,w,sample_weights,lambda_,sparse,loss_type,loss_fun,loss_gradient,conv_tol_heuristic,max_iter,heuristic_1,epsilon,max_iter_cg,max_iter_cg_heuristic)
	
	w,loss = _l2_svm_mfn(X,y,w,sample_weights,lambda_,sparse,loss_type,loss_fun,loss_gradient,conv_tol,max_iter,False if heuristic_2 else heuristic_1,epsilon,max_iter_cg,max_iter_cg_heuristic)
	return w,loss

#算法见论文Appendix A
def _l2_svm_mfn(X,y,w,sample_weights,lambda_,sparse,loss_type,loss_fun,loss_gradient,conv_tol,max_iter,heuristic_1,epsilon,max_iter_cg,max_iter_cg_heuristic):
	norm_X = norm_sparse(X) if sparse else norm(X)
	sqrt_sample_weights = None if sample_weights is None else np.sqrt(sample_weights)
	
	z = X.dot(w)
	nonzero_loss_indices = y*z<1
	zero_loss_indices = np.logical_not(nonzero_loss_indices)
	loss = loss_fun(y[nonzero_loss_indices],z[nonzero_loss_indices]) + 0.5*lambda_*np.sum(w**2)

	for iter_index in range(max_iter):
		if not np.any(nonzero_loss_indices):
			break

		nl_X,nl_y,nl_z = X[nonzero_loss_indices],y[nonzero_loss_indices],z[nonzero_loss_indices]
		nl_X_T = csr_matrix(nl_X.T) if sparse else np.array(nl_X.T,order='C')
		nl_sqrt_sample_weights = None if sqrt_sample_weights is None else sqrt_sample_weights[nonzero_loss_indices]

		w_new,optimality = cgls(w,lambda_,nl_X,nl_X_T,nl_y,nl_z,nl_sqrt_sample_weights,epsilon,norm_X,max_iter_cg_heuristic if iter_index==0 and heuristic_1 else max_iter_cg) 
		z_new = X.dot(w_new)
		
		zero_loss_indices = np.logical_not(nonzero_loss_indices)
		if optimality and np.all(y[nonzero_loss_indices]*z_new[nonzero_loss_indices]<1+conv_tol) and np.all(y[zero_loss_indices]*z_new[zero_loss_indices]>=1-conv_tol):
			w,z = w_new,z_new
			loss = loss_fun(y[nonzero_loss_indices],z[nonzero_loss_indices]) + 0.5*lambda_*np.sum(w**2)
			break
		
		w,z,nonzero_loss_indices,loss_ = line_search(X,y,sample_weights,lambda_,w_new,w,z_new,z,nonzero_loss_indices,loss_fun,loss)
		
		if loss_>loss:
			break
		loss = loss_
	
	return w,loss

