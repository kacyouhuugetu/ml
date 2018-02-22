from ..base import one_versus_rest_train,one_versus_one_train,one_versus_rest_predict,one_versus_one_predict,_classify_pre_process
from .bbr import BBR
from .cdn import CDN
from .newGLMNET import newGLMNET
from .tron import TRON
from .sgd import SGD
import numpy as np

class l1_logistic:
	def __init__(self,l1_algorithm,*algorithm_args,**algorithm_kwargs):
		"""
			l1 logistic算法接口。由于不同的l1 logistic算法有不同的算法结构，在这里只提供简单的方法。
			参数：
				①l1_algorithm：str，表示所使用的l1 logistic算法。当前可选有'bbr'、'cdn'、'glmnet'、'tron'和'sgd'
				②algorithm_args,algorithm_kwargs：l1 logistic算法所用参数
		"""

		if l1_algorithm == 'bbr':
			self._train_algorithm = BBR
		elif l1_algorithm == 'cdn':
			self._train_algorithm = CDN
		elif l1_algorithm == 'glmnet':
			self._train_algorithm = newGLMNET
		elif l1_algorithm == 'tron':
			self._train_algorithm = TRON
		elif l1_algorithm == 'sgd':
			self._train_algorithm = SGD
		else:
			raise ValueError("l1_algorithm must be 'bbr'、'cdn'、'glmnet'、'tron' or 'sgd'")
		
		self._l1_algorithm,self._l1_algorithm_args,self._l1_algorithm_kwargs = l1_algorithm,algorithm_args,algorithm_kwargs
	
	def train(self,X,y,C=1,w=None,sample_weights=None,sparse=False,conv_tol=1e-6,max_iter=100):
		"""
			训练算法求解解向量w，其使得目标函数f(w) = loss(w) + 0.5*lambda_*||w||^2最小
			参数：
				①X,y：训练集。y取值范围必须为{-1,1}
				②w：1D array或None，表示初始解向量。若w为None，则其为全为0的向量
				③sample_weights：1D array或None，表示各个数据对象的权重。若sample_weights为None，则认为所有数据对象具有相同权重(都为1)
				④sparse：bool，表示数据矩阵X是否为稀疏矩阵。默认为False
				⑤conv_tol：浮点数，表示算法收敛精度。默认为1e-3
				⑥max_iter：整型，表示算法的最大迭代次数。默认为100
		"""

		if w is None:
			w = np.zeros(X.shape[1],np.float64)
		self._w = self._train_algorithm(X,y,w,sample_weights,C,sparse,conv_tol,max_iter,*self._l1_algorithm_args,**self._l1_algorithm_kwargs)
		self._multi_class = False
	
	def multi_class_train(self,one_versus_one,n_class,X,y,C=1,sample_weights=None,sparse=False,conv_tol=1e-6,max_iter=100):
		"""
			利用one versus one或one versus rest训练多类l1 logistic。
			参数：
				①one_versus_one：bool，当one_versus_one为True，则使用one versus one训练多类SVM，否则使用one versus rest训练
				②n_class：int，表示类别的数量
				其余参数与train方法相同
		"""
		def train_method(X,y,*train_args):
			l1 = l1_logistic(self._l1_algorithm,*self._l1_algorithm_args,**self._l1_algorithm_kwargs)
			l1.train(X,y,*train_args)
			return l1

		train_args = (C,None,sample_weights,sparse,conv_tol,max_iter)
		
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
			def predict_method(X,l1):
				return l1.predict(X,return_score=self._multi_class_predict_algorithm==one_versus_rest_predict)
			
			return self._multi_class_predict_algorithm(X,self._n_class,self._train_res,predict_method)
		else:
			score = np.dot(X,self._w)
			return score if return_score else np.where(score>0,1,-1).astype(np.int8)
