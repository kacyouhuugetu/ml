from math import exp,log
from numpy.linalg import norm,slogdet,inv,eigh
from scipy.stats import multivariate_normal as mn
from sklearn.cluster import k_means
from ..myfun import _logpdf,logpdf
from ..base import less_equal
import numpy as np

class gmm:
	def __init__(self,K,cem=False,identity=False,diagonal=False,same_size=False,same_shape=False,same_direction=False):
		"""
			高斯混合模型(Gaussian Mixture Model)。高斯混合模型可以使用EM算法或其变种算法CEM算法来求解模型参数
			参数：
				①K：整型。表示高斯成分的数量
				②cem：bool。当cem为True，则使用CEM算法来求解高斯混合模型参数，否则使用EM算法来求解。默认为False
				③indentity,diagonal,same_size,same_shape,same_direction：bool。对各个高斯成分的方差的假设，具体作用见论文<Gaussian parsimonious clustering models>>
		"""
		self._K,self._cem = K,cem
		if cem:
			self._train_algorithm = gmm_cem
			#通过传入的参数确认高斯成分的方差类型
			#has_CF(algorithm_args[0])表示在CEM的M步时，高斯成分的方差的极大似然估计是否有闭式解
			self._algorithm_args = _gmm_cem_check(identity,diagonal,same_size,same_shape,same_direction) 
		else:
			self._train_algorithm,self._algorithm_args = gmm_em,()
	
	def train(self,X,init_mixture_weights=None,init_means=None,init_covs=None,outer_conv_tol=1e-6,outer_max_iter=100,inner_conv_tol=1e-6,inner_max_iter=100):
		"""
			使用EM/CEM算法求解高斯混合模型参数
			参数：
				①X：2D array。存储数据对象
				②init_mixture_weights,init_means,init_covs：高斯混合模型参数的初始解。默认为None，即不给出
				③outer_conv_tol：浮点数。表示EM/CEM算法收敛精度
				④outer_max_iter：整型。表示EM/CEM算法最大迭代次数
				⑤inner_conv_tol：浮点数。表示在CEM算法的M步时迭代求解方差的极大似然解时的收敛精度
				⑥inner_max_iter：整型。表示在CEM算法的M步时迭代求解方差的极大似然解时的最大迭代次数
		"""
		self._X = X
		self._mixture_weights,self._means,self._lambda,self._covs =  self._train_algorithm(X,self._K,init_mixture_weights,init_means,init_covs,outer_conv_tol,outer_max_iter,inner_conv_tol,inner_max_iter,*self._algorithm_args)
	
	def predict(self,new_X,return_log=True):
		"""
			给定数据，给出其各个高斯成分的概率密度。若return_log为True，则返回对数概率密度
		"""
		means,lambda_,covs = self._means,self._lambda,self._covs
		if self._cem:
			probs = _gmm_cem_calculate_logpdf(new_X,self._K,means,lambda_,covs,*self._algorithm_args)
			if not return_log:
				probs = np.log(probs,out=probs)
		else:
			pdf_fun = mn.logpdf if return_log else mn.pdf
			probs = np.stack([pdf_fun(new_X,mean,cov) for mean,cov in zip(means,covs)],axis=-1)
		
		return probs

	def get_model_args(self):
		"""
			返回高斯混合模型的参数
		"""
		if not hasattr(self,'_X'):
			raise ValueError('train first')

		mixture_weights,means,lambda_,covs = self._mixture_weights,self._means,self._lambda_,self._covs
		if self._cem:
			return mixture_weights,means,covs
		
		K,p = self._K,self._X.shape[1]
		has_CF,identity,diagonal,same_size,same_shape_same_direction,same_shape_diff_direction,diff_shape_diff_direction = self._algorithm_args

		if identity:
			covs = np.repeat(np.eye(p,dtype=np.float64)[np.newaxis],K,axis=0)
		elif diagonal:
			if same_shape_same_direction:
				covs = np.repeat(np.diag(covs if same_size else covs[0])[np.newaxis],K,axis=0)
			else:
				covs = np.stack([np.diag(cov) for cov in covs],axis=0)
		elif same_shape_same_direction and not same_size:
			covs[1:] = covs[0]

		if same_shape_diff_direction:
			for cluster_index in range(K):
				cov,coef = covs[cluster_index],lambda_ if same_size else lambda_[cluster_index]
				cov[:] = coef*np.dot(cov,cov.T)
		
		else:
			coef = lambda_ if same_size else  lambda_[:,np.newaxis,np.newaxis]
			covs *= coef
		
		return mixture_weights,means,covs

#求解高斯混合模型参数的初始解
def get_gmm_init_parameters(X,N,K,dtype=np.float64):
	_,label,_ = k_means(X,K)
	nk = np.bincount(label)
	init_mixture_weights = nk/N
	init_means = np.array([np.mean(X[label==cluster_index],axis=0) for cluster_index in range(K)],dtype)
	init_covs = np.array([np.cov(X[label==cluster_index],rowvar=False,ddof=0) for cluster_index in range(K)],dtype)
	
	return init_mixture_weights,init_means,init_covs

#EM算法具有很多变种算法，如CEM(Classification EM)算法和SEM(Stochastic EM)算法，二者较普通的EM算法有更快的收敛速度
#只需要稍微修改，gmm_em可以支持CEM算法和SEM算法(增加C步和S步)
#在这里只实现了普通的EM算法
def gmm_em(X,K,init_mixture_weights,init_means,init_covs,outer_conv_tol,outer_max_iter,inner_conv_tol,inner_max_iter):
	"""
		利用EM算法求解高斯混合模型(Gaussian Mixture Model)参数的极大似然解。利用EM算法求解高斯混合模型见论文<<A Gentle Tutorial of the EM Algorithm>>
		参数：
			①X：2D array。存储数据对象
			②K：整型。表示高斯密度成分(component)的数量
			③init_mixture_weights,init_means,init_covs：高斯混合模型参数的初始解。默认为None，即不给出
			④outer_conv_tol：浮点数。表示收敛精度
			⑤outer_max_iter：整型。表示算法最大迭代次数
	"""
	N = X.shape[0]
	if init_mixture_weights is None or init_means is None or init_covs is None:
		init_mixture_weights,init_means,init_covs = get_gmm_init_parameters(X,N,K)
	
	log_likelihood = -np.inf
	mixture_weights,means,covs = init_mixture_weights,init_means,init_covs
	for iter_index in range(outer_max_iter):
		#计算后验概率
		post_probs = np.array([mn.pdf(X,mean,cov) for mean,cov in zip(means,covs)],np.float64)*mixture_weights[:,np.newaxis]
		sum_post_probs = np.sum(post_probs,axis=0)
		
		#求解联立对数似然值，若两次迭代的对数似然值差值小于等于outer_conv_tol，则认为EM算法收敛
		log_likelihood_ = np.sum(np.log(sum_post_probs))
		if less_equal(abs(log_likelihood_-log_likelihood),outer_conv_tol):
			break
		
		log_likelihood = log_likelihood_
		post_probs /= sum_post_probs
		mixture_weights = np.sum(post_probs,axis=1)/N
		means = np.array([np.average(X,weights=prob_weights,axis=0) for prob_weights in post_probs],np.float64)
		covs = np.array([np.cov(X,rowvar=False,aweights=prob_weights,ddof=0) for prob_weights in post_probs],np.float64)
	return mixture_weights,means,None,covs

#在CEM算法的M步求解各个参数的极大似然解。根据不同的方差类型，求解方差的极大似然解的方法也不同
#不同类型的高斯成分方差类型的极大似然解求解见论文<<Gaussian parsimonious clustering models>>
def _gmm_cem_M_step(X,K,labels,mixture_weights,means,lambda_,covs,has_CF,identity,diagonal,same_size,same_shape_same_direction,same_shape_diff_direction,diff_shape_diff_direction,conv_tol,max_iter):
	N,p = X.shape
	if (identity or diff_shape_diff_direction) and same_size:
		lambda_ = 0.
	elif same_shape_diff_direction:
		eigvecs,eigvals = covs,np.empty((K,p),np.float64)
	elif same_shape_same_direction and same_size:
		covs[:] = 0.
	
	nk = np.empty(K,np.uint32)
	for cluster_index in range(K):
		cluster_indices = labels==cluster_index
		n_cluster_eles = np.sum(cluster_indices)
		X_cluster = X[cluster_indices]
		
		#在CEM中，某一数据对象所属的高斯成分的标号被看做一个参数而不是一个随机变量(EM算法将数据对象所属的高斯成分的标号视为随机变量)
		#nk[cluster_index]表示属于标号为cluster_index的高斯成分的数据对象的个数
		nk[cluster_index] = n_cluster_eles
		mixture_weights[cluster_index] = n_cluster_eles/N
		means[cluster_index] = np.mean(X_cluster,axis=0)
		
		#Wk为属于标号为cluster_index的高斯成分的数据对象的离方差，用于求解高斯成分方差的极大似然解
		Wk = np.cov(X_cluster,rowvar=False,ddof=n_cluster_eles-1)
		
		if diagonal:
			Wk = np.diag(Wk)
		if identity:
			if same_size:
				lambda_ += np.sum(Wk)/(N*p)
			else:
				lambda_[cluster_index] = np.sum(Wk)/(nk[cluster_index]*p)
		
		elif same_shape_diff_direction:
			eigvals[cluster_index],eigvecs[cluster_index] = eigh(Wk)
		
		else:
			if same_shape_same_direction and same_size:
				covs += Wk
			else:
				covs[cluster_index] = Wk

		if diff_shape_diff_direction:
			#当计算det(Wk)^(1/p)，我们并不直接求解det(Wk)，再求解其1/p幂
			#这是因为当Wk过大时，det(Wk)会发生上溢出
			#又由于det(Wk)^(1/p) = exp( logdet(Wk)/p )，我们首先计算logdet(Wk)/p，再求解其指数值
			#logdet(Wk)通常不会发生上溢出
			sign,logdet = (1,np.sum(np.log(Wk))) if diagonal else slogdet(Wk)
			det = sign*exp(logdet/p)
			covs[cluster_index] /= det
			if same_size:
				lambda_ += det/N
			else:
				lambda_[cluster_index] = det/nk[cluster_index]

	#has closed form
	if has_CF:
		if identity:
			pass
		elif same_shape_same_direction:
			covs /= N
			lambda_ = 1.
		elif same_shape_diff_direction:
			A = np.sum(eigvals,axis=0)
			log_detsum = np.sum(np.log(A))
			detsum = exp(log_detsum/p)
			covs *= np.sqrt(A/detsum)
			lambda_ = detsum/N

	else:
		#迭代求解same_size=False时,same_shape_same_direction或same_shape_diff_direction的最优解
		#在same_shape_same_direction中，C代表DAD'
		#在same_shape_diff_direction中，C代表A
		diagonal_eles = eigvals if same_shape_diff_direction else covs if diagonal else None
		C = np.sum(diagonal_eles,axis=0) if same_shape_diff_direction or diagonal else np.sum(covs,axis=0)

		#同样地，我们不直接计算det(C)^(1/p)，而是求解exp( logdet(C)/p )
		sign,logdet = (1,np.sum(np.log(C))) if same_shape_diff_direction or diagonal else slogdet(C)
		det = sign*exp(logdet/p)
		C /= det
		lambda_,obj = 1,np.inf
		for iter_index in range(max_iter):
			inv_C = np.reciprocal(C) if same_shape_diff_direction or diagonal else inv(C) 
			trace = np.sum(diagonal_eles*inv_C,axis=1) if same_shape_diff_direction or diagonal else np.sum(covs*inv_C.T,axis=(1,2)) 
			obj_ = np.sum(trace/lambda_) + np.sum(nk*np.log(lambda_))*p

			if less_equal(abs(obj-obj_),conv_tol):
				break
			
			obj = obj_
			lambda_ = trace/(nk*p)
			C = np.sum(diagonal_eles/lambda_[:,np.newaxis],axis=0) if diagonal or same_shape_diff_direction else np.sum(covs/lambda_[:,np.newaxis,np.newaxis],axis=0)
			
			#同样地，我们不直接计算det(C)^(1/p)，而是求解exp( logdet(C)/p )
			sign,logdet = (1,np.sum(np.log(C))) if same_shape_diff_direction or diagonal else slogdet(C)
			det = sign*exp(logdet/p)
			C /= det
		
		if same_shape_same_direction:
			covs[0] = C
		else:
			covs *= np.sqrt(C)
	return means,lambda_,covs

#根据参数决定高斯成分方差类型
def _gmm_cem_check(identity,diagonal,same_size,same_shape,same_direction):
	has_CF = True if same_size else False
	if identity:
		diagonal,same_shape,has_CF = True,True,True
	if diagonal:
		same_direction = True
	
	if same_shape:
		diff_shape_diff_direction = False
		same_shape_same_direction,same_shape_diff_direction = (True,False) if same_direction else (False,True)
	else:
		if same_direction and not diagonal:
			raise ValueError("don't support different shape with same direction currently")
		has_CF = True
		diff_shape_diff_direction = True
		same_shape_same_direction,same_shape_diff_direction = False,False

	return has_CF,identity,diagonal,same_size,same_shape_same_direction,same_shape_diff_direction,diff_shape_diff_direction

#根据高斯成分的方差类型，求解高斯成分的对数密度
def _gmm_cem_calculate_logpdf(X,K,means,lambda_,covs,has_CF,identity,diagonal,same_size,same_shape_same_direction,same_shape_diff_direction,diff_shape_diff_direction,out=None):
	log_probs = np.empty((X.shape[0],K),np.float64) if out is None else out
	p = X.shape[1]

	if identity:
		for cluster_index in range(K):
			log_probs[:,cluster_index] = _logpdf(X,means[cluster_index],None,1,p,coef=lambda_ if same_size else lambda_[cluster_index],identity=True)

	elif diagonal:
		if same_shape_same_direction:
			prec_U_ = np.sqrt(np.reciprocal(covs if same_size else covs[0]))
		for cluster_index in range(K):
			prec_U = prec_U_ if same_shape_same_direction else np.sqrt(np.reciprocal(covs[cluster_index]))
			log_probs[:,cluster_index] = _logpdf(X,means[cluster_index],prec_U,1,p,coef=lambda_ if same_size else lambda_[cluster_index],diagonal=True)

	elif same_shape_same_direction:
		cov = covs if same_size else covs[0]
		log_probs[:,0],psd = logpdf(X,means[0],cov,return_psd=True,coef=lambda_ if same_size else lambda_[0])

		for cluster_index in range(1,K):
			log_probs[:,cluster_index] = logpdf(X,means[cluster_index],cov,psd=psd,coef=lambda_ if same_size else lambda_[cluster_index])
	
	elif same_shape_diff_direction:
		for cluster_index in range(K):
			log_probs[:,cluster_index] = _logpdf(X,means[cluster_index],covs[cluster_index],1.,p,coef=lambda_ if same_size else lambda_[cluster_index])

	else:
		for cluster_index in range(K):
			log_probs[:,cluster_index] = logpdf(X,means[cluster_index],covs[cluster_index],coef=lambda_ if same_size else lambda_[cluster_index])
	
	return log_probs

#注意：当前只考虑协方差为对称正定的情形
def gmm_cem(X,K,init_mixture_weights=None,init_means=None,init_covs=None,outer_conv_tol=1e-6,outer_max_iter=100,inner_conv_tol=1e-6,inner_max_iter=100,identity=False,diagonal=False,same_size=False,same_shape=False,same_direction=False):
	"""
		利用CEM算法求解高斯混合模型(Gaussian Mixture Model)参数的极大似然解。CEM算法与利用CEM算法求解高斯混合模型见论文<<A classification EM algorithm for clustering and two stochastic versions>>、<<Gaussian parsimonious clustering models>>
		CEM算法较EM算法可以更快地达到收敛，又由于在E步时只需要求解对数密度，CEM算法具有更少的计算量
		当前本算法并不支持具有different shape(不同A)和same direction(相同D)的方差的极大似然求解
		参数：
			①X：2D array。存储数据对象
			②K：整型。表示高斯密度成分(component)的数量
			③init_mixture_weights,init_means,init_covs：高斯混合模型参数的初始解。默认为None，即不给出
			④outer_conv_tol：浮点数。表示CEM算法收敛精度
			⑤inner_conv_tol：浮点数。表示在CEM算法的M步时迭代求解方差的极大似然解时的收敛精度
			⑥outer_max_iter：整型。表示CEM算法最大迭代次数
			⑦inner_max_iter：整型。表示在CEM算法的M步时迭代求解方差的极大似然解时的最大迭代次数
	"""
	N,p = X.shape
	
	has_CF,identity,diagonal,same_size,same_shape_same_direction,same_shape_diff_direction,diff_shape_diff_direction = _gmm_cem_check(identity,diagonal,same_size,same_shape,same_direction)

	if init_mixture_weights is None or init_means is None or init_covs is None:
		init_mixture_weights,init_means,init_covs = get_gmm_init_parameters(X,N,K)
	mixture_weights,means,covs = init_mixture_weights,init_means,init_covs

	if same_size:
		lambda_ = 0.
	else:
		lambda_ = np.empty(K,np.float64)
	
	log_likelihood = -np.inf
	outer_iter_index = 0
	log_probs = np.empty((N,K),np.float64)
	while True:
		#在第一次迭代时，我们没有高斯成分的方差的用于传递给_gmm_cem_calculate_logpdf的信息
		#因此使用简单的方法直接求解对数密度
		if outer_iter_index == 0 :
			for cluster_index in range(K):
				log_probs[:,cluster_index] = logpdf(X,means[cluster_index],covs[cluster_index])
			if identity:
				covs = None
			elif same_shape_same_direction and same_size:
				covs = np.empty(p if diagonal else (p,p),np.float64)
			elif diagonal:
				covs = np.empty((K,p),np.float64)
		else:
			_gmm_cem_calculate_logpdf(X,K,means,lambda_,covs,has_CF,identity,diagonal,same_size,same_shape_same_direction,same_shape_diff_direction,diff_shape_diff_direction,out=log_probs)
		
		log_probs += np.log(mixture_weights)
		#在CEM算法中，数据对象所属的高斯成分的标号将被视为参数而非随机变量
		#因此，数据对象所属的高斯成分为具有最大后验概率的高斯成分
		labels = np.argmax(log_probs,axis=1)
		#求解极大似然值，若两次迭代的极大似然值差值小于等于outer_conv_tol，则认为CEM算法收敛
		log_likelihood_ = 0.
		for cluster_index in range(K):
			log_likelihood_ += np.sum(log_probs[labels==cluster_index])
		
		if outer_iter_index == outer_max_iter or less_equal(abs(log_likelihood_-log_likelihood),outer_conv_tol):
			break
		
		outer_iter_index += 1
		log_likelihood = log_likelihood_
		
		means,lambda_,covs = _gmm_cem_M_step(X,K,labels,mixture_weights,means,lambda_,covs,has_CF,identity,diagonal,same_size,same_shape_same_direction,same_shape_diff_direction,diff_shape_diff_direction,inner_conv_tol,inner_max_iter)
		
	return mixture_weights,means,lambda_,covs
