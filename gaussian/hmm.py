from math import log
from numpy.linalg import norm
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal as mn
from ..base import less_equal
from .gmm import get_gmm_init_parameters,_gmm_cem_check,_gmm_cem_calculate_logpdf,_gmm_cem_M_step
import numpy as np

class hmm:
	def __init__(self,M,mixture=False,zero_prob_tol=1e-6,*init_args,**init_kwargs):
		"""
			隐马尔科夫模型(Hidden Markov Model,HMM)，目前只支持observed distribution为高斯分布或高斯混合分布情形
			参数：
				①M：整型。状态的数量
				②mixture：bool。当mixture为True，则认为observed distribution为高斯混合分布，否则为高斯分布。默认为False
				③zero_prob_tol：浮点数。用于避免出现零概率的情形。当概率低于zero_prob_tol时，其将被置为zero_prob_tol。默认为1e-6
				③init_args,init_kwargs：传入初始化函数的参数。当用于高斯混合分布模型初始化时(mixture=True)，初始化函数的参数有：
					1)Ks：整型或整型的序列，表示各个混合高斯分布的高斯成分数量。当Ks为整型时，则所有高斯混合分布的高斯成分数量相同
					2)identity,diagonoal,same_size,same_shape,same_direction：bool或bool序列，表示对各个高斯混合分布的高斯成分的方差类型的假设，具体作用见论文<<Gaussian parsimonious clustering models>>
		"""
		self._M,self._mixture = M,mixture
		self._zero_prob_tol = zero_prob_tol
		self._init_args,self._init_kwargs = init_args,init_kwargs
		self._distribution_init,self._log_pdf_calculator,self._distribution_args_updator = (_gmm_cem_init,_gmm_cem_log_pdf_calculator,_gmm_cem_args_updator) if mixture else (_gaussian_distribution_init,_gaussian_log_pdf_calculator,_gaussian_distribution_args_updator)
	
	def train(self,X,conv_tol=1e-3,max_iter=100,warm_start=False):
		init_probs,trans_probs,*self._distribution_args = self._distribution_init(X,self._M,*self._init_args,**self._init_kwargs)
		self._log_init_probs,self._log_trans_probs = np.log(init_probs),np.log(trans_probs)
		self._log_observed_pdf = self._log_pdf_calculator(X,self._M,*self._distribution_args)
		self._X = X
		
		if not self._mixture and warm_start:
			self._segmental_K_means(conv_tol,max_iter)
		
		self._baum_welch(conv_tol,max_iter)

	def state_path(self):
		if not hasattr(self,'_X'):
			raise ValueError('train first')

		return self._viterbi()[0]
	
	#使用baum_welch算法求解hmm参数
	#当observed distribution为高斯分布或高斯混合分布时，hmm参数的求解可见论文<<A Gentle Tutorial of the EM Algorithm and its Application to Parameter Estimation for Gaussian Mixture and Hidden Markov Models>> 第4章
	#baum_welch通过计算expected number of transitions from state i ∑γ_t(i) = ∑p(st=i|O,λ)和expected number of transitions from state i to state j ∑ξ_t(i,j) = ∑p(st=i,st+1=j|O,λ)来求得hmm参数的极大似然估计
	#baum_welch算法可以看做EM算法在HMM上的应用，具体见论文
	def _baum_welch(self,conv_tol,max_iter):
		X,M,log_init_probs,log_trans_probs,log_observed_pdf,log_pdf_calculator,distribution_args_updator,distribution_args = self._X,self._M,self._log_init_probs,self._log_trans_probs,self._log_observed_pdf,self._log_pdf_calculator,self._distribution_args_updator,self._distribution_args
		T,ll = X.shape[0],-np.inf
		log_alpha,log_beta = np.empty((T,M),np.float64),np.empty((T,M),np.float64)

		for iter_index in range(max_iter):
			log_alpha,log_beta = self._forward(out=log_alpha),self._backward(out=log_beta)
			log_margin_prob = logsumexp(log_alpha[-1])
			log_state_i_post_probs = log_alpha + log_beta - log_margin_prob
			log_state_ij_post_probs = log_trans_probs + log_alpha[:-1][:,:,np.newaxis] + ( log_beta[1:] + log_observed_pdf[1:] )[:,np.newaxis,:] - log_margin_prob
			
			log_init_probs[:] = log_state_i_post_probs[0]
			log_trans_probs[:] = logsumexp(log_state_ij_post_probs,axis=0) - logsumexp(log_state_i_post_probs[:-1],axis=0)[:,np.newaxis]
			
			ll_observed_pdf = distribution_args_updator(X,M,log_observed_pdf,None,log_state_i_post_probs,*distribution_args)
			log_pdf_calculator(X,M,*distribution_args,out_log_pdf=log_observed_pdf)
			
			#ll_init_probs = logsumexp(log_state_i_post_probs[0],b=log_init_probs)
			#ll_trans_probs = logsumexp(log_state_ij_post_probs,b=log_trans_probs)
			ll_ = ll_observed_pdf #ll_init_probs + ll_trans_probs + ll_observed_pdf
			if less_equal(abs(ll-ll_),conv_tol):
				break
			ll = ll_

	#don't support mixture distribution currently
	#利用segmental_K_means求解hmm参数，见论文<<The Segmental K-Means Algorithm for Estimating Parameters of Hidden Markov Models>>
	#segmental_K_means像一般的K均值算法一样具有快速收敛的特点，因此可以作为其它算法的初始化算法
	#segmental_K_means求解hmm参数λ=argmax_λ {max_S p(S,O|λ)}
	#故算法首先利用viterbi算法求解使得p(S,O|λ)最大的state sequence S
	#又由max_S p(S,O|λ) = max_S logP(S,O|λ) = max_S {logp(O|S,λ) + logP(S|λ)}，故argmax_λ {max_S p(S,O|λ)}可以分解为argmax_λ {max_S logp(O|S,λ)} + argmax_λ {max_S logP(S|λ)}
	def _segmental_K_means(self,conv_tol,max_iter):
		zero_prob_tol,log_zero_prob_tol = self._zero_prob_tol,log(self._zero_prob_tol)
		X,M,log_init_probs,log_trans_probs,log_observed_pdf,log_pdf_calculator,distribution_args_updator,distribution_args = self._X,self._M,self._log_init_probs,self._log_trans_probs,self._log_observed_pdf,self._log_pdf_calculator,self._distribution_args_updator,self._distribution_args
		T,ll = X.shape[0],-np.inf

		for iter_index in range(max_iter):
			path,ll_,state_counts,trans_counts = self._viterbi()
			if less_equal(abs(ll-ll_),conv_tol):
				break
			
			ll = ll_
			log_init_probs[:] = log_zero_prob_tol
			log_init_probs[path[0]] = log(1-(M-1)*zero_prob_tol)
			
			for state_index in range(M):
				if state_counts[state_index] == 0:
					log_trans_probs[state_index] = -log(M)
				else:
					nz_trans_count_indices = trans_counts[state_index]!=0
					n_nz = np.sum(nz_trans_count_indices)
					log_trans_probs[state_index,nz_trans_count_indices] = np.log(trans_counts[state_index,nz_trans_count_indices] / state_counts[state_index] - (M-n_nz)/n_nz*zero_prob_tol) 
					log_trans_probs[state_index,np.logical_not(nz_trans_count_indices)] = log_zero_prob_tol
			
			state_indices = np.empty((T,M),np.bool8)
			for state_index in range(M):
				state_indices[:,state_index] = path==state_index
			
			distribution_args_updator(X,M,log_observed_pdf,state_indices,None,*distribution_args)
			log_pdf_calculator(X,M,*distribution_args,out_log_pdf=log_observed_pdf)

	#为了避免下溢出，我们并不直接计算alpha，而是计算log_alpha
	#在前向过程中，我们对log_alpha进行规格化以避免下溢出，具体见论文<<Hidden Markov Models>>(by Phil Blunsom)式(31)和式(32)
	def _forward(self,out=None):
		X,M,log_init_probs,log_trans_probs,log_observed_pdf = self._X,self._M,self._log_init_probs,self._log_trans_probs,self._log_observed_pdf
		T = X.shape[0]

		log_alpha = np.empty((T,M),np.float64) if out is None else out
		log_scale = np.empty(T,np.float64)

		log_alpha[0] = log_init_probs + log_observed_pdf[0]
		log_scale[0] = logsumexp(log_alpha[0])
		log_alpha -= log_scale[0]
		
		for t in range(1,T):
			log_alpha[t] = logsumexp(log_trans_probs + log_alpha[t-1][:,np.newaxis],axis=0) + log_observed_pdf[t]
			log_scale[t] = logsumexp(log_alpha[t])
			log_alpha[t] -= log_scale[t]

		log_scale = np.cumsum(log_scale,out=log_scale)
		log_alpha += log_scale[:,np.newaxis]
		return log_alpha

	#为了避免下溢出，我们并不直接计算beta，而是计算log_beta
	#在后向过程中，我们对log_alpha进行规格化以避免下溢出，具体见论文<<Hidden Markov Models>>(by Phil Blunsom)式(31)和式(32)
	def _backward(self,out=None):
		X,M,log_init_probs,log_trans_probs,log_observed_pdf = self._X,self._M,self._log_init_probs,self._log_trans_probs,self._log_observed_pdf
		T = X.shape[0]

		log_beta = np.empty((T,M),np.float64) if out is None else out
		log_scale = np.empty(T,np.float64)
		
		log_beta[-1] = -log(M)
		log_scale[-1] = log(M)

		for t in range(T-2,-1,-1):
			log_beta[t] = logsumexp( log_trans_probs + ( log_beta[t+1] + log_observed_pdf[t+1] ),axis=1)
			log_scale[t] = logsumexp(log_beta[t])
			log_beta[t] -= log_scale[t]
		
		np.cumsum(log_scale[::-1],out=log_scale[::-1])
		log_beta += log_scale[:,np.newaxis]
		return log_beta

	#求解使得后验联立分布p(S,O|λ)最大的state sequence S = s1,s2,...,sT
	#见论文<<An Introduction to Hidden Markov Models>>
	def _viterbi(self):
		M,log_init_probs,log_trans_probs,log_observed_pdf = self._M,self._log_init_probs,self._log_trans_probs,self._log_observed_pdf
		T = self._X.shape[0]
		cum_ll = np.empty((T,M),np.float64)
		
		cum_ll[0] = log_init_probs + log_observed_pdf[0]
		pre_states = np.empty((T-1,M),np.uint16)
		_indices_ = np.arange(M,dtype=np.uint16)
		
		for t in range(1,T):
			cum_ll_ = cum_ll[t-1][:,np.newaxis] + log_trans_probs
			max_index = np.argmax(cum_ll_,axis=0)
			pre_states[t-1] = max_index
			cum_ll[t] = cum_ll_[max_index,_indices_] + log_observed_pdf[t]
		
		path = np.empty(T,np.uint32)
		state = path[-1] = np.argmax(cum_ll[-1])
		ll = cum_ll[-1,state]
		state_counts,trans_counts = np.zeros(M,np.uint32),np.zeros((M,M),np.uint32)
		#令state = path[-1]
		#为了使得np.sum(trans_count,axis=1) == state_counts，我们令trans_counts[state,state]增加1，若不增加1，最终所得np.sum(trans_count[state]) == state_counts[state]-1
		state_counts[state] = trans_counts[state,state] = 1
		for t in range(T-2,-1,-1):
			pre_state = pre_states[t,state]
			trans_counts[pre_state,state] += 1
			state = pre_state
			path[t] = state
			state_counts[state] += 1

		return path,ll,state_counts,trans_counts

def _gaussian_distribution_init(X,M):
	init_probs,means,covs = get_gmm_init_parameters(X,X.shape[0],M)
	trans_probs = np.empty((M,M),np.float64)
	trans_probs[:] = 1./M

	return init_probs,trans_probs,means,covs

def _gaussian_log_pdf_calculator(X,M,means,covs,out_log_pdf=None):
	log_pdf = np.empty((X.shape[0],M),np.float64) if out_log_pdf is None else out_log_pdf
	for state_index in range(M):
		log_pdf[:,state_index] = mn.logpdf(X,means[state_index],covs[state_index])
	
	return log_pdf

def _gaussian_distribution_args_updator(X,M,logpdf,state_i_post_probs,log_state_i_post_probs,means,covs):
	if state_i_post_probs is None:
		state_i_post_probs = np.exp(log_state_i_post_probs)
	
	for state_index in range(M):
		means[state_index] = np.average(X,weights=state_i_post_probs[:,state_index],axis=0)
		covs[state_index] = np.cov(X,rowvar=False,aweights=state_i_post_probs[:,state_index],ddof=0)
	return np.sum(logpdf*state_i_post_probs)

#there are maybe some more efficient way to initialize hmm with gaussian mixture
def _gmm_cem_init(X,M,Ks,identity=False,diagonal=False,same_size=False,same_shape=False,same_direction=False,conv_tol=1e-3,max_iter=100):
	T = X.shape[0]
	
	#need more check
	Ks = np.repeat(Ks,M) if type(Ks)==int else np.array(Ks,np.uint8)
	identity = np.repeat(identity,M) if type(identity)==bool else np.array(identity,np.bool8)
	diagonal = np.repeat(diagonal,M) if type(diagonal)==bool else np.array(diagonal,np.bool8)
	same_size = np.repeat(same_size,M) if type(same_size)==bool else np.array(same_size,np.bool8)
	same_shape = np.repeat(same_shape,M) if type(same_shape)==bool else np.array(same_shape,np.bool8)
	same_direction = np.repeat(same_direction,M) if type(same_direction)==bool else np.array(same_direction,np.bool8)
	
	info = np.array([_gmm_cem_check(identity[state_index],diagonal[state_index],same_size[state_index],same_shape[state_index],same_direction[state_index]) for state_index in range(M)])
	
	NC = np.sum(Ks)
	mixture_weights,means,covs = get_gmm_init_parameters(X,T,NC)
	
	sorted_order = np.argsort(norm(means,axis=1))
	means = means[sorted_order]
	_indices_,unaccessed = np.arange(NC,dtype=np.uint32),np.ones(NC,np.bool8)
	order = []
	for state_index in range(M):
		index = np.nonzero(unaccessed)[0][0]
		order.append(index)
		unaccessed[index] = False
		
		dists = np.sum( (means[unaccessed] - means[index])**2 ,axis=1)
		sorted_indices = _indices_[unaccessed][np.argsort(dists)]
		unaccessed[sorted_indices[:Ks[state_index]-1]] = False
		order.extend(sorted_indices[:Ks[state_index]-1])
	
	sorted_order = sorted_order[order]
	cum_K = np.insert(np.cumsum(Ks),0,0)
	
	mixture_weights = np.array([mixture_weights[sorted_order[cum_K[state_index]:cum_K[state_index+1]]] for state_index in range(M)])
	means = np.array([means[order[cum_K[state_index]:cum_K[state_index+1]]] for state_index in range(M)])
	covs_ = np.empty(M,np.object)
	for state_index in range(M):
		covs_[state_index] = covs[sorted_order[cum_K[state_index]:cum_K[state_index+1]]]
	covs = covs_
	
	init_probs = np.array([np.sum(mixture_weights[state_index]) for state_index in range(M)],np.float64)
	trans_probs = np.empty((M,M),np.float64)
	trans_probs[:] = 1./M

	if np.all(Ks==Ks[0]):
		cluster_pdf = np.empty((M,T,Ks[0]),np.float64)
		if np.all(same_size==same_size[0]):
			lambdas = np.ones(M,dtype) if same_size[0] else np.ones((M,Ks[0]),np.float64)
	else:
		cluster_pdf = np.empty(M,dtype=np.object)
		for state_index in range(M):
			cluster_pdf[state_index] = np.empty((T,Ks[state_index]),np.float64)
		
		if np.all(same_size==True):
			lambdas = np.ones(M,np.float64)
		else:
			lambdas = np.empty(M,dtype=np.object)
			for state_index in range(M):
				lambdas[state_index] = 1 if same_size[state_index] else np.ones(Ks[state_index],np.float64)

	for state_index in range(M):
		cluster_means,cluster_covs,cluster_info = means[state_index],covs[state_index],info[state_index]
		for cluster_index in range(Ks[state_index]):
			cluster_pdf[state_index][:,cluster_index] = mn.logpdf(X,cluster_means[cluster_index],cluster_covs[cluster_index])
		
		#identity
		if cluster_info[1]:
			covs[state_index] = None
		#same_shape_same_direction and same_size
		elif cluster_info[3] and cluster_info[2]:
			covs[state_index] = np.empty(X.shape[1] if diagonal[state_index] else (X.shape[1],X.shape[1]),np.float64)

	return init_probs,trans_probs,Ks,info,mixture_weights,means,lambdas,covs,cluster_pdf,conv_tol,max_iter

def _gmm_cem_log_pdf_calculator(X,M,Ks,info,mixture_weights,means,lambdas,covs,cluster_pdf,conv_tol,max_iter,out_log_pdf=None):
	log_pdf = np.empty((X.shape[0],M),np.float64) if out_log_pdf is None else out_log_pdf

	for state_index in range(M):
		log_pdf[:,state_index] = logsumexp(cluster_pdf[state_index],b=mixture_weights[state_index],axis=1)
	
	return log_pdf

def _gmm_cem_args_updator(X,M,log_pdf,state_i_post_probs,log_state_i_post_probs,Ks,info,mixture_weights,means,lambdas,covs,cluster_pdf,conv_tol,max_iter):
	ll = 0.
	for state_index in range(M):
		cluster_pdf[state_index] += np.log(mixture_weights[state_index])
		log_state_i_cluster_j_post_probs = cluster_pdf[state_index] \
										+ log_state_i_post_probs[:,state_index][:,np.newaxis] \
										- log_pdf[:,state_index][:,np.newaxis]
		
		labels = np.argmax(log_state_i_cluster_j_post_probs,axis=1)
		for cluster_index in range(Ks[state_index]):
			ll += logsumexp(cluster_pdf[state_index][labels==cluster_index])

		Ks_,mixture_weights_,means_,lambdas_,covs_,info_ = Ks[state_index],mixture_weights[state_index],means[state_index],lambdas[state_index],covs[state_index],info[state_index]

		_gmm_cem_M_step(X,Ks[state_index],labels,mixture_weights_,means_,lambdas_,covs_,*info_,conv_tol=conv_tol,max_iter=max_iter)
		_gmm_cem_calculate_logpdf(X,Ks_,means_,lambdas_,covs_,*info_,out=cluster_pdf[state_index])

	return ll

