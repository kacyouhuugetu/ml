from ..base import linear_kernel,gaussian_kernel,one_versus_rest_train,one_versus_one_train,one_versus_rest_predict,one_versus_one_predict,_classify_pre_process
from .svm_base import _LOSS_TYPE_SVM_L1_,_LOSS_TYPE_SVM_L2_,_LOSS_TYPE_LOGISTIC_,_loss_svm_l1,_loss_svm_l2,_loss_logistic,_loss_svm_l1_gradient,_loss_svm_l2_gradient,_loss_logistic_gradient
from .pegasos import pegasos
from .trust_region import trust_region
from .cmls import cmls
from .mfn import mfn
from .cd import cdsvm,cddual
from .smo_svm import SMO
from .multi_svm import multisvm
import numpy as np

class linear_svm:
	def __init__(self,svm_algorithm,loss='l1',*algorithm_args,**algorithm_kwargs):
		"""
			linear svm算法接口。由于不同的linear svm算法有不同的算法结构，在这里只提供简单的方法。
			参数：
				①svm_algorithm：str，表示所使用的svm算法。当前可选有'pegasos'、'tr'、'cmls'、'mfn'、'cdsvm'和'cddual'
				②loss：str，表示所使用的损失函数。当前可选有'l1'表示Hinge-Loss、'l2'表示Squared Hinge-Loss和'logistic'表示logistic loss
				③algorithm_args,algorithm_kwargs：svm算法所用参数
		"""

		if loss=='l1':
			loss_type,loss_function,loss_gradient = _LOSS_TYPE_SVM_L1_,_loss_svm_l1,_loss_svm_l1_gradient
		elif loss=='l2':
			loss_type,loss_function,loss_gradient = _LOSS_TYPE_SVM_L2_,_loss_svm_l2,_loss_svm_l2_gradient
		elif loss=='logistic':
			loss_type,loss_function,loss_gradient = _LOSS_TYPE_LOGISTIC_,_loss_logistic,_loss_logistic_gradient
		else:
			raise ValueError("loss must be 'l1'、'l2' or 'logistic'")

		if svm_algorithm == 'pegasos':
			self._train_algorithm = pegasos
		elif svm_algorithm == 'tr':
			if loss_type == _LOSS_TYPE_SVM_L1_:
				raise ValueError('trust region algorithm does not support l1 loss')
			self._train_algorithm = trust_region
		elif svm_algorithm == 'cmls':
			if loss_type == _LOSS_TYPE_SVM_L1_:
				raise ValueError('cmls algorithm does not support l1 loss')
			self._train_algorithm = cmls
		elif svm_algorithm == 'mfn':
			if loss_type != _LOSS_TYPE_SVM_L2_:
				raise ValueError('mfn algorithm only support l2 loss')
			self._train_algorithm = mfn
		elif svm_algorithm == 'cdsvm':
			if loss_type != _LOSS_TYPE_SVM_L2_:
				raise ValueError('cdsvm algorithm only support l2 loss')
			self._train_algorithm = cdsvm
		elif svm_algorithm == 'cddual':
			if loss_type == _LOSS_TYPE_LOGISTIC_:
				raise ValueError('cddual algorithm does not support logistic loss')
			self._train_algorithm = cddual
		else:
			raise ValueError("svm_algorithm must be 'pegasos'、'tr'、'cmls'、'mfn'、'cdsvm' or 'cddual'")
		
		self._loss,self._loss_type,self._loss_function,self._loss_gradient = loss,loss_type,loss_function,loss_gradient
		self._svm_algorithm,self._svm_algorithm_args,self._svm_algorithm_kwargs = svm_algorithm,algorithm_args,algorithm_kwargs

	def train(self,X,y,lambda_,sample_weights=None,init_vector=None,sparse=False,conv_tol=1e-6,max_iter=100):
		"""
			训练算法求解解向量w，其使得目标函数f(w) = loss(w) + 0.5*lambda_*||w||^2最小
			参数：
				①X,y：训练集。y取值范围必须为{-1,1}
				②init_vector：1D array或None，表示初始解向量。若svm_algorithm为cddual，则init_vector为初始拉格朗日乘子α。若init_vector为None，则其为全为0的向量
				③sample_weights：1D array或None，表示各个数据对象的权重。若sample_weights为None，则认为所有数据对象具有相同权重(都为1)
				④sparse：bool，表示数据矩阵X是否为稀疏矩阵。默认为False
				⑤conv_tol：浮点数，表示算法收敛精度。默认为1e-3
				⑥max_iter：整型，表示算法的最大迭代次数。默认为100
		"""
		init_vector = np.zeros(X.shape[0] if self._train_algorithm == cddual else X.shape[1],np.float64)
		
		self._w,_ = self._train_algorithm(X,y,init_vector,sample_weights,lambda_,sparse,self._loss_type,self._loss_function,self._loss_gradient,conv_tol,max_iter,*self._svm_algorithm_args,**self._svm_algorithm_kwargs)

		self._multi_class = False

	def multi_class_train(self,one_versus_one,n_class,X,y,lambda_,sample_weights=None,sparse=False,conv_tol=1e-6,max_iter=100):
		"""
			利用one versus one或one versus rest训练多类SVM。
			参数：
				①one_versus_one：bool，当one_versus_one为True，则使用one versus one训练多类SVM，否则使用one versus rest训练
				②n_class：int，表示类别的数量
				其余参数与train方法相同
		"""
		def train_method(X,y,*train_args):
			svm = linear_svm(self._svm_algorithm,self._loss,*self._svm_algorithm_args,**self._svm_algorithm_kwargs)
			svm.train(X,y,*train_args)
			return svm

		train_args = (lambda_,sample_weights,None,sparse,conv_tol,max_iter)
		
		self._multi_class_train_algorithm,self._multi_class_predict_algorithm = (one_versus_one_train,one_versus_one_predict) if one_versus_one else (one_versus_rest_train,one_versus_rest_predict)
		self._train_res = self._multi_class_train_algorithm(X,y,train_method,train_args)
		
		self._n_class,self._multi_class = n_class,True

	def predict(self,X,return_score=False):
		"""
			预测
		"""
		if not hasattr(self,'_multi_class'):
			raise ValueError('train first')
		
		if self._multi_class:
			def predict_method(X,svm):
				return svm.predict(X,return_score=self._multi_class_predict_algorithm==one_versus_rest_predict)
			
			return self._multi_class_predict_algorithm(X,self._n_class,self._train_res,predict_method)
		else:
			score = np.dot(X,self._w)
			return score if return_score else np.where(score>0,1,-1).astype(np.int8)

class kernel_svm:
	def __init__(self,svm_algorithm='smo',loss='l1',kernel='linear',kernel_args=(),compute_kernel=False,buf_ratio=1.0,*algorithm_args,**algorithm_kwargs):
		"""
			kernel svm算法接口。由于不同的kernel svm算法有不同的算法结构，在这里只提供简单的方法。
			参数：
				③kernel：str或callable object，表示核函数。当kernel为str，其可选为'linear'(线性核)或'gaussian'(高斯核)。当kernel为callable object，其接受两个array X1和X2，其中X2为shape为(n,p)的2D array。当X1为shape为(m,p)的2D array时，函数返回shape为(m,n)的2D array；若X1为shape为(p,)的1D array，则函数返回shape为(n,)的1D array。默认为'gaussian'
				④kernel_args：元组，表示传入kernel的参数。
				⑤svm_l1_loss：bool，表示loss的类型。若svm_l1_loss为True，则为Hinge-Loss，否则为Squared Hinge-Loss。默认为True
				⑥compute_kernel：bool，表示是否计算并存储核矩阵。计算并存储核矩阵可以提高算法效率，但需要更多的存储空间。默认为False
				⑦buf_ratio：浮点数，表示计算核矩阵时的批处理量。默认为1.0
		"""
		
		if loss == 'l1':
			self._svm_l1_loss = True
		elif loss == 'l2':
			self._svm_l1_loss = False
		else:
			raise ValueError("loss must be 'l1' or 'l2'")
		self._loss = loss

		if kernel == 'linear':
			self._kernel = linear_kernel
		elif kernel == 'gaussian':
			self._kernel = gaussian_kernel
		elif callable(kernel):
			self._kernel = kernel
		else:
			raise ValueError("kernel must be str like 'linear' or 'gaussian',or callable object")

		if svm_algorithm == 'smo':
			self._train_algorithm = SMO
		elif svm_algorithm == 'multisvm':
			self._train_algorithm = multisvm
		else:
			raise ValueError("svm_algorithm must be 'smo'")

		self._kernel_args,self._compute_kernel,self._buf_ratio = kernel_args,compute_kernel,buf_ratio
		self._svm_algorithm,self._svm_algorithm_args,self._svm_algorithm_kwargs = svm_algorithm,algorithm_args,algorithm_kwargs
	
	def train(self,X,y,C,sample_weights=None,alpha=None,tol=1e-6,max_iter=100,n_class=None):
		self._K = self._kernel(X,X,*self._kernel_args) if self._compute_kernel else None
		if self._svm_algorithm == 'smo':
			self._alpha,self._b_low,self._b_up = self._train_algorithm(X,y,alpha,sample_weights,C,self._kernel,self._kernel_args,self._svm_l1_loss,tol,max_iter,self._K,self._buf_ratio,*self._svm_algorithm_args,**self._svm_algorithm_kwargs)
		else:
			if n_class is None:
				raise ValueError('n_class should be integer')
			self._alpha = self._train_algorithm(alpha,X,y,sample_weights,C,self._kernel,self._kernel_args,self._svm_l1_loss,tol,max_iter,self._K,self._buf_ratio,n_class,*self._svm_algorithm_args,**self._svm_algorithm_kwargs)

		self._X,self._y = X.copy(),y.copy()

	def predict(self,new_X,return_score=False):
		if not hasattr(self,'_X'):
			raise ValueError('train first')

		N = new_X.shape[0]
		X = self._X
		kernel,kernel_args = self._kernel,self._kernel_args
		is_smo = self._svm_algorithm == 'smo'
		
		if is_smo:
			alpha,b = self._alpha*self._y,(self._b_low+self._b_up)/2
		else:
			alpha,b = self._alpha,0.0

		if self._K is None:
			buf_size = min(N,max(1,int(self._buf_ratio*N)))
			predict = np.empty(N,np.int8 if is_smo else np.uint32)
			
			start_index,end_index = 0,buf_size
			while start_index<N:
				k = kernel(new_X[start_index:end_index],X,*kernel_args)
				score = np.dot(k,alpha) + b
				if is_smo:
					predict[start_index:end_index] = np.where(score>0,1,-1).astype(np.int8)
				else:
					predict[start_index:end_index] = np.argmax(score,axis=1)
				start_index,end_index = end_index,end_index+buf_size

		else:
			score = np.dot(self._K,alpha) + b
			if is_smo:
				predict = np.where(score>0,1,-1).astype(np.int8)
			else:
				predict = np.argmax(score,axis=1)
	
		return predict
