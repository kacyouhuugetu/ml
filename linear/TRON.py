from math import sqrt
from numpy.linalg import norm
from ..base import less_equal
import numpy as np

#计算海森矩阵与d的矩阵乘法
def hessian_dot(X_T_pos,X_T_neg,X_pos,X_neg,D,d,n_pos_free,hess_dot,d_index=None):
	if d_index is None:
		d_index = n_pos_free
	hess_dot[:n_pos_free] = X_T_pos.dot(D*X_pos.dot(d[:d_index])) - X_T_pos.dot(D*X_neg.dot(d[d_index:]))
	hess_dot[n_pos_free:] = -X_T_neg.dot(D*X_pos.dot(d[:d_index])) + X_T_neg.dot(D*X_neg.dot(d[d_index:]))

#共轭梯度算法求解论文式(53)，见论文Algorithm 7
def CG(X_T_pos,X_T_neg,X_pos,X_neg,D,gradient_free,gradient_qd_free,hess_dot_d,d,n_pos_free,region_size,tol):
	b = -gradient_qd_free
	x,pre_x = np.zeros_like(d),np.zeros_like(d)
	r,p = b.copy(),b.copy() 
	norm2_square_r = np.sum(r**2)
	stop_criterion = tol**2*np.sum(gradient_free**2)
	
	A_dot_p = np.empty_like(d)
	for _ in range(d.shape[0]):
		hessian_dot(X_T_pos,X_T_neg,X_pos,X_neg,D,p,n_pos_free,A_dot_p)
		alpha = norm2_square_r/np.dot(p,A_dot_p)
		
		x += alpha*p
		if norm(x+d)>region_size:
			a,b,c = np.sum(p**2),2*np.dot(pre_x+d,p),np.sum((pre_x+d)**2)-region_size**2
			delta = b**2-4*a*c
			gamma = max((-b+sqrt(delta))/(2*a),(-b-sqrt(delta))/(2*a)) if delta>=0 else 0
			return pre_x+gamma*p,True
		pre_x = x.copy()
		
		r -= alpha*A_dot_p
		norm2_square_r_ = np.sum(r**2)
		if less_equal(norm2_square_r_,stop_criterion):
			break

		beta = norm2_square_r_/norm2_square_r
		p *= beta
		p += r
		norm2_square_r = norm2_square_r_
	else:
		return x,True
	return x,False


def TRON(X,y,w,sample_weights,C,sparse,conv_tol,max_iter,sigma=0.01,beta=0.9,eta0=1e-3,eta1=0.25,eta2=0.75,sigma1=0.25,sigma2=0.5,sigma3=4.0,max_region_size=1e10,zero_atol=1e-10):
	"""
		TRON算法。详细见论文<<A Comparison of Optimization Methods and Software for Large-scale L1-regularized Linear Classification>> 5.1节
		参数：
			①eta0,eta1,eta2,sigma1,sigma2,sigma3：浮点数。TRON算法用于权向量和region size的更新，具体见论文
			②max_region_size：浮点数，表示最大的region size。
	"""
	N,p = X.shape
	if sparse:
		X = csr_matrix(X)
		X_T = csr_matrix(X.T)
	else:
		X_T = X.T
	
	w_ = np.zeros(p*2,np.float64)
	if not w is None:
		pos_indices,neg_indices = w>0,w<0 
		w_[:p][pos_indices] = w[pos_indices]
		w_[p:][neg_indices] = w[neg_indices]
	w = w_
	
	C = C if sample_weights is None else C*sample_weights

	exp_neg_Xy_dot_w = np.exp(-y*X.dot(w[:p]-w[p:]))
	
	#计算C*[σ(yw'x)-1]
	probs_minus_one = (np.reciprocal(1+exp_neg_Xy_dot_w)-1)*C
	#计算C*{ σ(yw'x)*[1-σ(yw'x)] }
	D = np.reciprocal(exp_neg_Xy_dot_w+np.reciprocal(exp_neg_Xy_dot_w)+2.)*C
	#实际梯度为一长度为2p的向量，在这里我们只计算前p个元素组成的向量gradient
	#易得，实际梯度后p个元素组成的向量为-X_T.dot(probs_minus_one*y) + 1，即2-gradient
	gradient = X_T.dot(probs_minus_one*y) + 1
	
	loss = np.sum(C*np.log(exp_neg_Xy_dot_w+1)) + np.sum(w)
	#初始化∆为实际梯度的2-范数，见论文<<Newton's method for large-scale bound constrained problems>>
	region_size= sqrt( np.sum(gradient**2) + np.sum((2-gradient)**2) )
	
	cauchy_d = np.empty(p*2,np.float64)
	for iter_index in range(max_iter):
		lambda_ = 1
		#利用线性搜索来计算cauchy point，见论文式(47)-(49)
		while True:
			cauchy_d[:p] = np.clip(w[:p] - lambda_*(gradient),0,None) - w[:p]
			cauchy_d[p:] = np.clip(w[p:] - lambda_*(2-gradient),0,None) - w[p:]
			#计算实际梯度与cauchy_d的点积
			gradient_dot_d = np.dot(gradient,cauchy_d[:p]) + np.dot(2-gradient,cauchy_d[p:])
			#计算实际海森矩阵于cauchy_d的矩阵乘法
			#令H=X'DX，易得实际海森矩阵Hessian为
			#             |	H   -H |
			#   Hessian = |        |
			#             |	-H   H |
			#hess_dot_d = np.dot(| H   -H |,cauchy_d)
			#易得，np.dot(Hessian,cauchy_d) = np.dot(cauchy_d[:p],hess_dot_d) - np.dot(cauchy_d[p:],hess_dot_d)
			hess_dot_d = X_T.dot(D*X.dot(cauchy_d[:p])) - X_T.dot(D*X.dot(cauchy_d[p:]))
			qd = 0.5* ( np.dot(cauchy_d[:p],hess_dot_d) - np.dot(cauchy_d[p:],hess_dot_d) ) + gradient_dot_d
			condition = gradient_dot_d
			
			#寻找满足qd <= sigma*condition且||cauchy_d||<=region_size的步长
			if less_equal(qd,sigma*condition) and less_equal(norm(cauchy_d),region_size):
				break
			lambda_ *= beta

		w_,d = w+cauchy_d,cauchy_d
		#计算F(cauchy_point)
		free_indices = np.logical_not(np.isclose(w_,0,atol=zero_atol))
		n_pos_free = np.sum(free_indices[:p])
		
		gradient_free = np.hstack((gradient[free_indices[:p]],2-gradient[free_indices[p:]]))
		X_T_pos,X_T_neg = X_T[free_indices[:p]],X_T[free_indices[p:]]
		X_pos,X_neg = ( csr_matrix(X_T_pos.T),csr_matrix(X_T_neg.T) ) if sparse else (X_T_pos.T,X_T_neg.T)
		hess_dot_d = np.hstack((hess_dot_d[free_indices[:p]],-hess_dot_d[free_indices[p:]]))
		gradient_qd_free = hess_dot_d + gradient_free
		for t in range(2*p):
			if not np.any(free_indices):
				d = w_ - w
				break
			
			v,early_stop = CG(X_T_pos,X_T_neg,X_pos,X_neg,D,gradient_free,gradient_qd_free,hess_dot_d,d[free_indices],n_pos_free,region_size,conv_tol)
			lambda_ = 1
			#线性搜索，见论文式(55)
			while True:
				d_ = d.copy()
				w_free = np.clip(w_[free_indices] + lambda_*v,0,None)
				d_[free_indices] = w_free - w[free_indices]
				
				gradient_dot_d_ = np.dot(gradient,d_[:p]) + np.dot(2-gradient,d_[p:])
				hess_dot_d_ = X_T.dot(D*X.dot(d_[:p])) - X_T.dot(D*X.dot(d_[p:]))
				qd_ = 0.5* ( np.dot(d_[:p],hess_dot_d_) - np.dot(d_[p:],hess_dot_d_) ) + gradient_dot_d_ 
				
				condition = np.dot(gradient_qd_free,d_[free_indices] - d[free_indices])
				if less_equal(qd_-qd,sigma*condition):
					w_[free_indices],d,qd = w_free,d_,qd_
					free_indices = np.logical_not(np.isclose(w_,0,atol=zero_atol))
					n_pos_free = np.sum(free_indices[:p])
					gradient_free = np.hstack((gradient[free_indices[:p]],2-gradient[free_indices[p:]]))
					X_T_pos,X_T_neg = X_T[free_indices[:p]],X_T[free_indices[p:]]
					X_pos,X_neg = ( csr_matrix(X_T_pos.T),csr_matrix(X_T_neg.T) ) if sparse else (X_T_pos.T,X_T_neg.T)
					hess_dot_d = np.empty(np.sum(free_indices),np.float64)
					hessian_dot(X_T_pos,X_T_neg,X,X,D,d,n_pos_free,hess_dot_d,p)
					gradient_qd_free = hess_dot_d + gradient_free
					break

				lambda_ *= beta
			
			if less_equal(norm(gradient_qd_free),conv_tol*norm(gradient_free)) or early_stop:
				d = w_ - w
				break
		
		new_w = w + d

		exp_neg_Xy_dot_w = np.exp(-y*X.dot(new_w[:p]-new_w[p:]))
		new_loss = np.sum(C*np.log(exp_neg_Xy_dot_w+1)) + np.sum(new_w)
		#计算reduction ratio，见论文式(45)
		rho = (new_loss - loss)/qd
		#更新权向量，见论文式(46)
		if rho>eta0:
			w,loss = new_w,new_loss
			D = np.reciprocal(exp_neg_Xy_dot_w+np.reciprocal(exp_neg_Xy_dot_w)+2.)*C
			probs_minus_one = (np.reciprocal(1+exp_neg_Xy_dot_w)-1)*C
			gradient = X_T.dot(probs_minus_one*y) + 1
		
		#更新region size
		if region_size<max_region_size:
			region_size = (sigma1*min(norm(d),region_size)+sigma2*region_size)/2 if less_equal(rho,eta1) else\
						(1+sigma3)*region_size/2 if less_equal(eta2,rho) else \
						(sigma1+sigma3)*region_size/2 

	return w[:p]-w[p:]
