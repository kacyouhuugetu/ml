from math import sqrt,log
from scipy.linalg import cholesky,solve_triangular
from scipy.stats import norm as normal
from scipy.optimize import minimize
from ..base import logistic_fun,log_logistic_fun,get_norm2_square,gaussian_kernel,standard_normal_pdf,standard_normal_cdf,less_equal
import numpy as np

#x is scalar
#five_cdf(x) = standard_normal_cdf(x*LAMBDAS) = [Φ(x*λ_1),...,Φ(x*λ_5)]
#我们可以对sigmoid函数作近似，即有σ(x) ≈ np.dot( COEFs, five_cdf(x) )
#这样，∫σ(x)*N(x|μ，C)dx ≈ ∑ COEFi * ∫Φ(x*λi)*N(x|μ，C)dx,其中N(x|μ，C)表示均值为μ，方差为C的高斯密度
#而∫Φ(x*λi)*N(x|μ，C)dx可以很简单地求出，见维基百科：
#https://en.wikipedia.org/wiki/Error_function#Integral_of_error_function_with_Gaussian_density_function
#或见<<Gaussian Processes for Machine Learning>>式(3.82)
LAMBDAs = np.array([0.57982756,0.56568542,0.52325902,0.62225397,0.55154329],np.float64)
COEFs = np.array([-1854.8214151,3516.89893646,221.29346712,128.12323805,-2010.49422654],np.float64)

#使用BFGS算法求解最优theta值
def get_theta_BFGS(method,init_theta,X,y,trans_y,norm2_square,response_distribution,_indices_,**method_args):
	init_theta = np.array((init_theta,),np.float64)
	theta_gradient = np.empty_like(init_theta)
	args = (X,y,trans_y,norm2_square,response_distribution,_indices_,theta_gradient)
	
	method = laplace_approximation if method=='laplace' else ep 
	obj = lambda theta,*args:-method(theta,*args,eval_theta_gradient=True,**method_args)
	jac = lambda theta,*args:-args[-1]
	obj(init_theta,*args)
	res = minimize(obj,init_theta,args,method='L-BFGS-B',jac=jac,bounds=((0,None),))
	return res.x[0]

#使用穷举的方法在区间[min_theta,max_theta]上求解最优theta值
def get_theta_region_search(method,min_theta,max_theta,n_search,log_scale,X,y,trans_y,norm2_square,response_distribution,_indices_,**method_args):
	#若log_scale=False,则使用线性搜索，否则在对数搜索
	thetas = np.exp(np.linspace(log(min_theta),log(max_theta),n_search)) if log_scale else np.linspace(min_theta,max_theta,n_search)

	max_ll,max_theta = -np.inf,None
	method = laplace_approximation if method=='laplace' else ep
	for theta in thetas:
		ll = method(theta,X,y,trans_y,norm2_square,response_distribution,_indices_,return_log_margin=True,**method_args)
		if ll>max_ll:
			max_ll,max_theta = ll,theta

	return max_theta

#laplace近似求解f的后验分布，见书籍<<Gaussian Processes for Machine Learning>>算法3.1
#theta_gradient、eval_theta_gradient、return_log_margin用于求解log marginal likelihood logq(y|X,θ)对theta的梯度，在求解最优theta时(使对数似然值最大)使用
def laplace_approximation(theta,X,y,trans_y,norm2_square,response_distribution,_indices_=None,theta_gradient=None,eval_theta_gradient=False,return_log_margin=False,conv_tol=1e-3,zero_prob_tol=1e-6,max_iter=100):
	N = X.shape[0]
	if eval_theta_gradient:
		K,K_derivative = gaussian_kernel(X,X,theta[0],norm2_square,return_derivative=True)
	else:
		K = gaussian_kernel(X,X,theta,norm2_square)

	B,sqrt_W = np.empty((N,N),np.float64),np.empty((N,N),np.float64)
	
	a = f = np.zeros(N,np.float64)
	_indices_ = np.arange(N,dtype=np.uint32) if _indices_ is None else _indices_
	
	iter_index,obj = -1,-np.inf
	while True:
		#计算对数条件概率logp(y|f)及对数似然值
		#当response_distribution == 'logistic'，则logp(y|f) = -log(1+exp(-y*f))
		#当response_distribution == 'CGaussian'，则logp(y|f) = logΦ(y*f)
		#zero_prob_tol用于求解p(y=1|f)时出现0概率情形，若p(y=1|f)<zero_prob_tol，则p(y=1|f)将被置为zero_prob_tol
		if response_distribution == 'logistic':
			probs = logistic_fun(f)
			probs[probs<zero_prob_tol] = zero_prob_tol
			log_likelihood = np.sum(log_logistic_fun(y*f))
		else:
			cdf_yf = standard_normal_cdf(y*f)
			log_likelihood = np.sum(np.log(cdf_yf))

		obj_ = -0.5*np.dot(a,f) + log_likelihood
		if less_equal(abs(obj_-obj),conv_tol) or iter_index==max_iter:
			break
		
		iter_index += 1
		obj = obj_
		
		#求解对数条件概率logp(y|f)对f的梯度和二阶梯度，见式(3.13)-(3.16)
		if response_distribution == 'logistic':
			log_likelihood_gradient,W = trans_y - probs , probs*(1-probs)
		else:
			y_mul_pdf_f_div_cdf_yf = y*standard_normal_pdf(f) / cdf_yf
			log_likelihood_gradient,W = y_mul_pdf_f_div_cdf_yf , y_mul_pdf_f_div_cdf_yf*(y_mul_pdf_f_div_cdf_yf+f)
		
		#由于f满足i.i.d条件，logp(y|f)对f的信息矩阵为对角矩阵，W为该信息矩阵的对角元素
		diag_sqrt_W = np.sqrt(W)
		#B = I + W^(1/2)*K*W^(1/2)，*为矩阵乘法，见式(3.26)
		B[:] = K
		B *= diag_sqrt_W
		B *= diag_sqrt_W[:,np.newaxis]
		B[_indices_,_indices_] += 1
		
		sqrt_W[:] = 0.
		sqrt_W[_indices_,_indices_] = diag_sqrt_W
		#算法3.1第5行。L为矩阵B的cholesky分解，为下三角矩阵
		L = B[:] = cholesky(B,lower=True,overwrite_a=True,check_finite=False)
		#L_div_sqrt_W为L^(-1)*W^(1/2)，*为矩阵乘法
		L_div_sqrt_W = sqrt_W[:] = solve_triangular(L,sqrt_W,lower=True,overwrite_b=True,check_finite=False)
		L_div_sqrt_W_gram = L_div_sqrt_W[:] = np.dot(L_div_sqrt_W.T,L_div_sqrt_W)
		
		#算法3.1第6行
		b = f*W + log_likelihood_gradient
		#算法3.1第7行
		b -= np.dot(L_div_sqrt_W_gram,np.dot(K,b))
		a = b
		#算法3.1第8行
		f = np.dot(K,a)
	
	if eval_theta_gradient or return_log_margin:
		#算法3.1第10行，计算log marginal likelihood logq(y|X,θ)
		log_margin_likelihood = obj - np.sum(np.log(np.diag(L)))
		
		#计算log marginal likelihood logq(y|X,θ)对theta的梯度，见算法5.1
		if eval_theta_gradient:
			#计算条件概率密度logp(y|f)对f的三阶梯度
			if response_distribution == 'logistic':
				third_derivative = W*(1-2*probs)
			else:
				g = y_mul_pdf_f_div_cdf_yf
				third_derivative = -2*g**3 + f*g**2 + f**2*g + g
			
			#算法5.1第7-9行
			Z = L_div_sqrt_W
			diag_C_gram = np.sum(np.dot(K,L_div_sqrt_W_gram)*K,axis=1)
			s2 = -0.5*(np.diag(K)-diag_C_gram)*third_derivative
			
			#算法5.1第11-14行
			C = K_derivative
			s1 = 0.5*( np.dot(a,np.dot(C,a)) - np.sum(C*Z) )
			b = np.dot(C,log_likelihood_gradient)
			s3 = b - np.dot(K,np.dot(Z,b))
			
			#算法5.1第15行
			theta_gradient[:] = s1 + np.dot(s2,s3)
		return log_margin_likelihood

	else:
		return log_likelihood_gradient,L_div_sqrt_W_gram

#ep近似求解f的后验分布，见书籍<<Gaussian Processes for Machine Learning>>算法3.5。ep算法见书籍<<A family of algorithms for approximate Bayesian inference>>
#theta_gradient、eval_theta_gradient、return_log_margin用于求解log marginal likelihood logq(y|X,θ)对theta的梯度，在求解最优theta时(使target分布对数似然值最大)使用
def ep(theta,X,y,trans_y,norm2_square,response_distribution,_indices_=None,theta_gradient=None,eval_theta_gradient=False,return_log_margin=False,conv_tol=1e-3,max_iter=100):
	N = X.shape[0]
	if eval_theta_gradient:
		K,K_derivative = gaussian_kernel(X,X,theta[0],norm2_square,return_derivative=True)
	else:
		K = gaussian_kernel(X,X,theta,norm2_square)

	approx_post_mean,approx_post_var = np.zeros(N,np.float64),K.copy()
	approx_ll_v,approx_ll_precision = np.zeros_like(approx_post_mean),np.zeros_like(approx_post_mean)
	leave_one_out_mean,leave_one_out_precision = np.zeros_like(approx_post_mean),np.zeros_like(approx_post_mean)
	B,sqrt_S = np.empty((N,N),np.float64),np.empty((N,N),np.float64)
	_indices_ = np.arange(N,dtype=np.uint32) if _indices_ is None else _indices_
	
	iter_index,sum_log_cdf_z = 0,0.
	while True:
		new_sum_log_cdf_z = 0.
		all_converge = True
		for i in range(N):
			approx_ll_precision_i,approx_ll_v_i = approx_ll_precision[i],approx_ll_v[i]
			approx_post_margin_mean_i,approx_post_margin_precision_i = approx_post_mean[i],1./approx_post_var[i,i]
			#算法3.5第5、6行
			leave_one_out_precision_i = approx_post_margin_precision_i - approx_ll_precision_i
			leave_one_out_v_i = approx_post_margin_precision_i*approx_post_margin_mean_i - approx_ll_v_i
			#calculate the parameters of post marginal distribution
			y_i,leave_one_out_var_i = y[i],1./leave_one_out_precision_i
			leave_one_out_mean_i = leave_one_out_var_i*( approx_post_margin_precision_i*approx_post_margin_mean_i - approx_ll_v_i )
			
			#算法第7行，利用式(3.58)计算margin moments
			#书籍中只给出了response_distribution为cumulative gaussian时，margin moments的计算方法
			#通过对logistic函数进行近似，可以简单地得到response_distribution为logistic函数时的margin moments
			if response_distribution == 'logistic':
				v = y_i/(LAMBDAs)
				z_denominator = v*np.sqrt(1.+leave_one_out_var_i/v**2)
				z = leave_one_out_mean_i/z_denominator 
				pdf_z,cdf_z = standard_normal_pdf(z),standard_normal_cdf(z)

				Z = np.dot(COEFs,cdf_z)
				post_margin_mean_i = leave_one_out_mean_i \
								+ leave_one_out_var_i*np.dot(COEFs,pdf_z/z_denominator)/Z
				post_margin_var_i = leave_one_out_var_i \
								- leave_one_out_var_i**2 * np.dot(COEFs,z*pdf_z/z_denominator**2)/Z \
								- ( post_margin_mean_i - leave_one_out_mean_i )**2
				new_sum_log_cdf_z += log(Z)
			else:
				sqrt_1_plus_var = sqrt(1.+leave_one_out_var_i)
				z = (y_i*leave_one_out_mean_i)/sqrt_1_plus_var
				pdf_z,cdf_z = standard_normal_pdf(z),standard_normal_cdf(z)
				pdf_div_cdf = pdf_z/cdf_z

				new_sum_log_cdf_z += log(cdf_z)
				post_margin_mean_i = leave_one_out_mean_i \
								+ pdf_div_cdf*( y_i*leave_one_out_var_i/sqrt_1_plus_var )
				post_margin_var_i = leave_one_out_var_i \
							- pdf_div_cdf* leave_one_out_var_i**2/(1.+leave_one_out_var_i) \
							* ( z+pdf_div_cdf )
			post_margin_precision_i = 1./post_margin_var_i

			#算法3.5第8、9行
			delta_approx_ll_precision_i = post_margin_precision_i - leave_one_out_precision_i - approx_ll_precision_i
			delta_approx_ll_v_i = post_margin_precision_i*post_margin_mean_i - leave_one_out_v_i - approx_ll_v_i
			new_approx_ll_precision_i,new_approx_ll_v_i = approx_ll_precision_i+delta_approx_ll_precision_i,approx_ll_v_i+delta_approx_ll_v_i
			
			#若对第i个样本，两次迭代的Δγ和Δv变化差值小于conv_tol，则认为算法对第i个样本收敛
			if less_equal(abs(approx_ll_precision_i-new_approx_ll_precision_i),conv_tol) and less_equal(abs(approx_ll_v_i-new_approx_ll_v_i),conv_tol):
				continue

			all_converge = False
			approx_ll_precision[i],approx_ll_v[i] = new_approx_ll_precision_i,new_approx_ll_v_i
			leave_one_out_mean[i],leave_one_out_precision[i] = leave_one_out_mean_i,leave_one_out_precision_i
			
			#update parameters of post distribution
			#Δ∑ = sign*np.outer(si,si)
			coef = 1./(1./delta_approx_ll_precision_i + approx_post_var[i,i])
			coef,sign = (sqrt(coef),-1) if coef>0 else (sqrt(-coef),1)
			si = coef*approx_post_var[:,i]
			
			#算法3.5第11行
			#经过算法前10行计算后，有更新为∑' = ∑ + Δ∑，v' = v + Δv
			#则μ' = ∑'*v' = (∑+Δ∑)*(v+Δv) = ∑v + ∑*Δv + Δ∑v' = μ + Δμ
			#其中，Δ∑*v' = sign*np.dot(si,np.dot(si,v'))
			#又由于算法每次只对一个样本的v进行更新，故Δv除第i个元素外所有元素都为0，则∑*Δv = si*(Δv[i]/coef)
			delta_approx_post_mean = si*( sign*( np.dot(si,approx_ll_v) ) + (delta_approx_ll_v_i/coef) )
			approx_post_mean += delta_approx_post_mean

			#equivalent to post_var -= np.outer(si,si)
			#算法3.5第10行，更新方差
			approx_post_var /= si
			if sign>0:
				approx_post_var += si[:,np.newaxis]
			else:
				approx_post_var -= si[:,np.newaxis]
			approx_post_var *= si

		#若对所有样本，两次迭代的Δγ和Δv变化差值小于conv_tol，则认为算法收敛
		#一个更好的做法是计算ZEP(算法5.1第17行)来判断是否收敛
		if all_converge or iter_index==max_iter:
			break
		sum_log_cdf_z = new_sum_log_cdf_z
		iter_index += 1
		
		#计算B，见式(3.67)
		diag_sqrt_S = np.sqrt(approx_ll_precision)
		B[:] = K
		B *= diag_sqrt_S
		B *= diag_sqrt_S[:,np.newaxis]
		B[_indices_,_indices_] += 1
		
		sqrt_S[:] = 0
		sqrt_S[_indices_,_indices_] = diag_sqrt_S
		#算法3.5第13行
		L = B[:] = cholesky(B,lower=True,overwrite_a=True,check_finite=False)
		L_div_sqrt_S = sqrt_S[:] = solve_triangular(L,sqrt_S,lower=True,overwrite_b=True,check_finite=False)
		L_div_sqrt_S_gram = L_div_sqrt_S[:] = np.dot(L_div_sqrt_S.T,L_div_sqrt_S)
		
		Kv,V_gram = np.dot(K,approx_ll_v),np.dot(K,np.dot(L_div_sqrt_S_gram,K),out=approx_post_var)
		#算法3.5第15行
		approx_post_mean[:] = Kv - np.dot(V_gram,approx_ll_v)
		approx_post_var *= -1
		approx_post_var += K


	if eval_theta_gradient or return_log_margin:
		inv_S_plus_T = np.reciprocal(approx_ll_precision + leave_one_out_precision)
		approx_post_var[_indices_,_indices_] -= inv_S_plus_T
		
		#算法3.5第17行，利用式(3.65)、(3.73)和(3.74)计算logZEP
		log_Zep_fourth_and_first_term = 0.5*np.sum(np.log(1+approx_ll_precision/leave_one_out_precision)) - np.sum(np.log(np.diag(L)))
		log_Zep_part_fifth_and_second_term = 0.5*np.dot(approx_ll_v,np.dot(approx_post_var,approx_ll_v))
		log_Zep_remainder_fifth_term = 0.5*np.dot(leave_one_out_mean,\
											leave_one_out_precision*inv_S_plus_T\
											*(approx_ll_precision*leave_one_out_mean-2*approx_ll_v)
											)
		log_Zep_third_term = sum_log_cdf_z
		log_Zep = log_Zep_fourth_and_first_term + log_Zep_part_fifth_and_second_term + log_Zep_remainder_fifth_term + log_Zep_third_term
		
		#计算log marginal likelihood logq(y|X,θ)对theta的梯度，见算法5.2
		if eval_theta_gradient:
			#算法5.2第5、6行
			b = approx_ll_v - np.dot(L_div_sqrt_S_gram,np.dot(K,approx_ll_v))
			Z = -L_div_sqrt_S_gram.copy()
			Z /= b
			Z += b[:,np.newaxis]
			Z *= b
			
			#算法5.2第9行
			theta_gradient[:] = 0.5*np.sum(K_derivative*Z.T)
		
		return log_Zep

	else:
		#算法3.6第3行
		v_minus_z = approx_ll_v - np.dot(L_div_sqrt_S_gram,Kv)
		return v_minus_z,L_div_sqrt_S_gram

#GPC:Gaussian Process for Classification
#only support RBF kernel currently
class gpc:
	def __init__(self,theta,response_distribution='logistic',method='laplace',conv_tol=1e-3,zero_prob_tol=1e-6,max_iter=100,search_theta=False,search_method='gradient',min_theta=None,max_theta=None,n_search=None,log_scale=False,buf_size=None,buf_ratio=1.):
		"""
			GPC算法(Gaussian Process for Classification)。目前只用于实现二类分类，Covariance Function只可选为RBF核。详细见书籍<<Gaussian Processes for Machine Learning>>第3章和第5章
			本算法采用laplace近似和ep近似来求解latent function f的后验分布，laplace近似见书籍3.4节，ep近似见书籍3.6节
			theta作为RBF核的参数，其对gpc结果的影响是巨大的。为了求解"最优"(使对数似然值log marginal likelihood logq(y|X,θ)最大)的theta，算法可以选择使用BFGS或穷举的方法来求解最优的theta
			参数：
				①theta：RBF核的参数。K(x,x') = exp(-θ*||x-x'||^2)
				②response_distribution：str，表示target分布。当前可选为'logistic'表示target分布为logistic分布，或'CGaussian'表示Cumulative Gaussian分布。默认为'logistic'
				③method：str，表示对latent function f所作的高斯近似。'laplace'表示使用laplace近似，'ep'表示使用expectation propagation近似
				④conv_tol：浮点数，算法的收敛精度
				⑤max_iter：整型，算法的最大迭代次数
				⑥search_theta：bool，是否搜索最优的theta值
				⑦search_method：str，表示最优theta的搜索方法。只有当search_theta=True时有效。若search_method为'gradient'，则将使用theta的梯度信息，将参数①theta作为初始解，使用BFGS算法来求解最优theta。否则，将使用穷举的方法来求解最优theta，穷举的区间，穷举的值个数由min_theta,max_theta,n_search来指定。search_method默认为'gradient'
				⑧min_theta,max_theta,n_search：若search_theta为True且search_method不为'gradient'，将使用穷举的方法来求解最优theta。穷举的区间为[min_theta,max_theta]，穷举的值个数为n_seach
				⑨log_scale：bool，表示穷举theta值时，区间的搜索方法。若log_scale=False，则使用线性搜索，否则使用对数搜索。默认为False
		"""

		if method=='laplace':
			self._method_args = {'conv_tol':conv_tol,'zero_prob_tol':zero_prob_tol,'max_iter':max_iter}
		else:
			self._method_args = {'conv_tol':conv_tol,'max_iter':max_iter}
		
		if search_theta and search_method!='gradient':
			if min_theta is None or max_theta is None or n_search is None:
				raise ValueError("min_theta、max_theta and n_search can't be None")
			if min_theta<=0 or max_theta<=0 or n_search<=0:
				raise ValueError("min_theta、max_theta and n_search should be all larger than 0")
			if min_theta>=max_theta:
				raise ValueError("min_theta larger equal than max_theta")
		
		self._theta,self._response_distribution,self._method,self._search_theta,self._search_method,self._min_theta,self._max_theta,self._n_search,self._log_scale,self._buf_size,self._buf_ratio = theta,response_distribution,method,search_theta,search_method,min_theta,max_theta,n_search,log_scale,buf_size,buf_ratio
		
	def train(self,X,y):
		self._X = X
		trans_y = (y+1)//2
		if self._search_theta:
			norm2_square = get_norm2_square(X,X)
			_indices_ = np.arange(X.shape[0],dtype=np.uint32)
			if self._search_method == 'gradient':
				self._theta = get_theta_BFGS('laplace',self._theta,X,y,trans_y,norm2_square,self._response_distribution,_indices_,**self._method_args)
			else:
				self._theta = get_theta_region_search('laplace',self._min_theta,self._max_theta,self._n_search,self._log_scale,X,y,trans_y,norm2_square,self._response_distribution,_indices_,**self._method_args)
		else:
			norm2_square,_indices_ = None,None
		
		if self._method == 'laplace':
			log_likelihood_gradient,L_div_sqrt_W_gram = laplace_approximation(self._theta,X,y,trans_y,norm2_square,self._response_distribution,_indices_,**self._method_args)
			self._f_means_factor,self._f_vars_factor = log_likelihood_gradient,L_div_sqrt_W_gram
		else:
			v_minus_z,L_div_sqrt_S_gram = ep(self._theta,X,y,trans_y,norm2_square,self._response_distribution,_indices_,**self._method_args)
			self._f_means_factor,self._f_vars_factor = v_minus_z,L_div_sqrt_S_gram
	
	def predict(self,new_X):
		X,theta,response_distribution,buf_size = self._X,self._theta,self._response_distribution,self._buf_size
		f_means_factor,f_vars_factor = self._f_means_factor,self._f_vars_factor
		N,M = X.shape[0],new_X.shape[0]

		if buf_size is None:
			buf_ratio = max(0,min(self._buf_ratio,1))
			buf_size = max(1,int(M*buf_ratio))
		
		probs = np.empty(M,np.float64)
		start_index,end_index = 0,buf_size
		while start_index<M:
			#由于使用gaussian_kernel，kk为1
			k,kk = gaussian_kernel(new_X[start_index:end_index],X,theta),1
			f_means = np.dot(k,f_means_factor)
			f_vars = kk - np.sum(k*np.dot(k,f_vars_factor),axis=1)
			
			#当response_distribution为'logistic'，需要将其近似为五个累积正态分布的加权和
			if response_distribution == 'logistic':
				v = 1./(LAMBDAs)[:,np.newaxis]
				z = f_means/(v*np.sqrt(1+f_vars/v**2))
				probs[start_index:end_index] = np.dot(COEFs,standard_normal_cdf(z))

			else:
				probs[start_index:end_index] = standard_normal_cdf(f_means/np.sqrt(1+f_vars)) 
			start_index,end_index = end_index,end_index+buf_size

		return probs
