from math import sqrt
from numpy.linalg import norm
from scipy.sparse import csr_matrix,isspmatrix_csr
from .svm_base import _LOSS_TYPE_SVM_L1_,_LOSS_TYPE_SVM_L2_,_LOSS_TYPE_LOGISTIC_,_get_nl
from ..base import isclose
import numpy as np

def pegasos(X,y,w,sample_weights,lambda_,sparse,loss_type,loss_fun,loss_gradient,conv_tol,max_iter,learning_rate='scale',sample_size_ratio=1.0,projection_step=False):
	"""
		pegasos算法。见论文<<Pegasos: Primal Estimated sub-GrAdient SOlver for SVM>>
		参数：
			①learning_rate：str或callable object，用于计算学习速率η。当learning rate为callable object，其接受参数k和lambda_，并返回learning rate η，其中参数k表示当前迭代步数。当learning rate为str，可选为'scale'(η=1/(k*lambda_))。默认为'scale'
			②sample_size_ratio：浮点数，表示每次迭代时样本的批处理量
			③projection_step：bool，表示是否使用projection。若使用projection，则解向量w的2-范数将被限制不超过1/sqrt(lambda_)。默认为False
	"""

	N,p = X.shape
	if isspmatrix_csr(X):
		sparse = True
	elif sparse:
		X = csr_matrix(X)
	
	indices = np.arange(N)
	n_sample = min(N,max(1,int(N*sample_size_ratio)))
	svm_loss = (loss_type==_LOSS_TYPE_SVM_L1_) or (loss_type==_LOSS_TYPE_SVM_L2_)
	sqrtlambda_ = sqrt(lambda_)
	
	if learning_rate == 'scale':
		learning_rate = lambda k,lambda_:1./(k*lambda_)
	elif not callable(learning_rate):
		raise ValueError("learning_rate should be set to 'linear' or callable object")

	for iter_index in range(1,max_iter+1):
		eta = learning_rate(iter_index,lambda_)
		sample_indices = np.random.choice(indices,replace=False,size=n_sample)
		x_,y_,sample_weights_ = X[sample_indices],y[sample_indices],None if sample_weights is None else sample_weights[sample_indices]
		
		nl_x,nl_y,nl_z,nl_sample_weights,_ = _get_nl(w,x_,y_,sample_weights_,svm_loss)
		loss = loss_fun(nl_y,nl_z,nl_sample_weights) + 0.5*lambda_*np.sum(w**2)

		v = loss_gradient(nl_x.T,nl_y,nl_z,nl_sample_weights)
		sub_gradient = lambda_*w + v
		w_ = w - eta*sub_gradient
		
		#见论文式(6)
		if projection_step and lambda_>0:
			radius = 1./(sqrtlambda_*norm(w_))
			if radius<1:
				w_ *= radius
		
		nl_x,nl_y,nl_z,nl_sample_weights,_ = _get_nl(w_,x_,y_,sample_weights_,svm_loss)
		loss_ = loss_fun(nl_y,nl_z,nl_sample_weights) + 0.5*lambda_*np.sum(w_**2)

		if isclose(abs(loss-loss_),conv_tol) and isclose((norm(w)-norm(w_)),conv_tol):
			break
		w = w_
	
	nl_x,nl_y,nl_z,nl_sample_weights,_ = _get_nl(w,X,y,sample_weights,svm_loss)
	loss = loss_fun(nl_y,nl_z,nl_sample_weights) + 0.5*lambda_*np.sum(w**2)

	return w,loss
