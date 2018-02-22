from numpy.linalg import norm
from scipy.sparse import csr_matrix,isspmatrix_csr
from .svm_base import _LOSS_TYPE_SVM_L1_,_LOSS_TYPE_SVM_L2_,_LOSS_TYPE_LOGISTIC_,_get_nl
from ..base import isclose,logistic_fun
import numpy as np

#计算Hv，其中H为目标函数的hessian矩阵
def compute_hess_dot(X,X_T,d,sample_weights,lambda_,svm_loss,preconditioned,inv_P,D):
	if preconditioned:
		inv_P_dot_d = inv_P*d
		if svm_loss:
			if sample_weights is None:
				hess_dot_d = inv_P * ( lambda_*inv_P_dot_d + X_T.dot(X.dot(inv_P_dot_d)) )
			else:
				hess_dot_d = inv_P * ( lambda_*inv_P_dot_d + X_T.dot(sample_weights*X.dot(inv_P_dot_d)) )
		else:
			hess_dot_d = inv_P * ( lambda_*inv_P_dot_d + X_T.dot(D*X.dot(inv_P_dot_d)) )
	else:
		if svm_loss:
			if sample_weights is None:
				hess_dot_d = lambda_*d + X_T.dot(X.dot(d)) 
			else:
				hess_dot_d = lambda_*d + X_T.dot(sample_weights*X.dot(d)) 
		else:
			hess_dot_d = lambda_*d + X_T.dot(D*X.dot(d))

	return hess_dot_d

#见论文Algorithm 2
def cg_trust_region(X,X_T,sample_weights,gradient,lambda_,svm_loss,preconditioned,inv_P,D,xi,region_size,max_iter):
	N = X.shape[0]
	xi**=2
	
	D = D if svm_loss or sample_weights is None else D*sample_weights
	s,pre_s = np.zeros_like(gradient),np.zeros_like(gradient)
	r = -gradient.copy()
	d = r.copy()
	norm2_square_r = np.sum(r**2)
	norm2_square_gradient = np.sum(gradient**2)
	for iter_index in range(1,max_iter+1):
		if norm2_square_r<=xi*norm2_square_gradient:
			return s
		
		pre_s[:] = s
		hess_dot_d = compute_hess_dot(X,X_T,d,sample_weights,lambda_,svm_loss,preconditioned,inv_P,D)
		alpha = norm2_square_r/np.dot(hess_dot_d,d)
		s += alpha*d
		
		if norm(s)>=region_size:
			a,b,c = np.sum(d**2),2*np.dot(pre_s,d),np.sum(pre_s**2)-region_size**2
			delta = b**2-4*a*c
			gamma = max((-b+sqrt(delta))/(2*a),(-b-sqrt(delta))/(2*a)) if delta>=0 else 0
			return pre_s+gamma*d
		
		r -= alpha*hess_dot_d
		norm2_square_r_ = np.sum(r**2)
		beta = norm2_square_r_/norm2_square_r
		norm2_square_r = norm2_square_r_
		d*=beta
		d+=r

	return s

def _get_nl_(w,X,y,sample_weights,svm_loss,X_power,sparse,preconditioned):
	nl_X,nl_y,nl_z,nl_sample_weights,nonzero_loss_indices  = _get_nl(w,X,y,sample_weights,svm_loss)
	nl_X_power = None

	if preconditioned:
		if svm_loss:
			nl_X_power = X_power[nonzero_loss_indices] if not X_power is None else nl_X.power(2) if sparse else nl_X**2
		else:
			nl_X_power = X_power if not X_power is None else nl_X.power(2) if sparse else nl_X**2
	
	return nl_X,nl_y,nl_z,nl_sample_weights,nl_X_power,nonzero_loss_indices

def trust_region(X,y,w,sample_weights,lambda_,sparse,loss_type,loss_fun,loss_gradient,conv_tol,max_iter,xi=0.1,eta0=1e-4,eta1=0.25,eta2=0.75,sigma1=0.25,sigma2=0.5,sigma3=4.0,compute_power=False,preconditioned=True,max_iter_cg=100):
	"""
		trust region for l2-regularization。见论文<<Trust Region Newton Method for Large-Scale Logistic Regression>>
		参数：
			①xi,eta0,eta1,eta2,sigma1,sigma2,sigma3：浮点数。trust region算法参数，具体见论文
			②compute_power：bool，表示是否计算并存储X**2。存储X**2可以提高算法效率，但需要更多的存储空间
			③preconditioned：bool，表示求解牛顿步时是否使用preconditioned CG算法。若preconditioned为False，则使用普通的CG算法求解牛顿步。默认为True
			④max_iter_cg：int，表示用CG算法求解牛顿步时，最大的迭代次数
	"""
	N,p = X.shape
	if isspmatrix_csr(X):
		sparse = True
	elif sparse: 
		X = csr_matrix(X)
	
	if preconditioned and compute_power:
		X_power = X.power(2) if sparse else X**2 
	else:
		X_power = None

	svm_loss = loss_type==_LOSS_TYPE_SVM_L2_
	
	nl_X,nl_y,nl_z,nl_sample_weights,nl_X_power,_ = _get_nl_(w,X,y,sample_weights,svm_loss,X_power,sparse,preconditioned)
	nl_X_T = csr_matrix(nl_X.T) if sparse else np.array(nl_X.T,order='C')
	
	D = 0
	loss = np.inf
	region_size = norm(loss_gradient(nl_X_T,nl_y,nl_z))
	
	for iter_index in range(1,max_iter+1):
		inv_P = None

		if svm_loss:
			if preconditioned:
				sum_nl_X_power = np.ravel(nl_X_power.sum(axis=0)) if sparse else np.sum(nl_X_power,axis=0)
				inv_P = np.reciprocal(np.sqrt(lambda_ + sum_nl_X_power))
		else:
			probs = logistic_fun(nl_y*nl_z)
			D = probs*(1-probs)
			if preconditioned:
				sum_X_power_prod_D = nl_X_power.T.dot(D)
				inv_P = np.reciprocal(np.sqrt(lambda_ + sum_X_power_prod_D))
		
		gradient = lambda_*w + loss_gradient(nl_X_T,nl_y,nl_z,nl_sample_weights)
		if np.max(np.abs(gradient))<=conv_tol or isclose(region_size,0,1e-10):
			break
		
		if preconditioned:
			gradient *= inv_P
		
		#使用CG算法计算牛顿步，见论文2.2节和Algorithm 2
		s = cg_trust_region(nl_X,nl_X_T,nl_sample_weights,gradient,lambda_,svm_loss,preconditioned,inv_P,D,xi,region_size,max_iter_cg)
		hess_dot_s = compute_hess_dot(nl_X,nl_X_T,s,nl_sample_weights,lambda_,svm_loss,preconditioned,inv_P,D)
		#计算二次模型，即计算f(w_new)-f(w)的二次泰勒展开
		q = np.dot(gradient,s) + 0.5*np.dot(s,hess_dot_s)
		
		if preconditioned:
			s_k = inv_P*s
		w_new = w + ( s_k if preconditioned else s )
		
		nl_X_new,nl_y_new,nl_z_new,nl_sample_weights_new,_,nonzero_loss_indices = _get_nl_(w_new,X,y,sample_weights,svm_loss,X_power,sparse,False)
		
		loss_ = 0.5*lambda_*np.sum(w_new**2) + loss_fun(nl_y_new,nl_z_new,nl_sample_weights_new)
		if isclose(q,0):
			rho = np.inf if loss_<loss else -np.inf
		else:
			rho = (loss_-loss)/q
		
		#更新trust region，见论文式(8)-式(10)
		if rho>eta0:
			w = w_new
			nl_X,nl_y,nl_z,nl_sample_weights = nl_X_new,nl_y_new,nl_z_new,nl_sample_weights_new
			if preconditioned:
				if svm_loss:
					nl_X_power = X_power[nonzero_loss_indices] if compute_power else nl_X.power(2) if sparse else nl_X**2
				else:
					nl_X_power = X_power if compute_power else nl_X.power(2) if sparse else nl_X**2
			nl_X_T = csr_matrix(nl_X.T) if sparse else np.array(nl_X.T,order='C')
			loss = loss_
		
		if rho<=eta1:
			region_size = ( sigma1*min(norm(s),region_size) + sigma2*region_size ) /2
		elif rho>=eta2:
			region_size = ( region_size + sigma3*region_size) /2
		else:
			region_size = ( sigma1*region_size + sigma3*region_size ) /2
		
	loss = 0.5*lambda_*np.sum(w**2) + loss_fun(nl_y,nl_z,nl_sample_weights)
	return w,loss
			
