from ..base import isclose,less_equal,linear_kernel,gaussian_kernel
import numpy as np

#求解loss为Hinge-Loss时的sub-problem，具体见论文<<On the Learnability and Design of Output Codes for Multiclass Problems>> Figure 3
def sub_problem_solve_search_l1(A,B,C,n_class,class_):
	D = B/A
	D[class_] += C
	sorted_indices = np.argsort(-D)
	D_ = D[sorted_indices]

	r=0
	fai,pre_fai = C,0
	while fai>0 and r<n_class-1:
		r += 1
		pre_fai = fai
		fai += r*(D_[r]-D_[r-1])
	if fai>0:
		r += 1
	else:
		fai = pre_fai

	theta = D_[r-1] - fai/r
	D[sorted_indices[:r]] = theta

	return D - B/A

#求解loss为Squared Hinge-Loss时的sub-problem，具体见论文<<A Study on L2-Loss (Squared Hinge-Loss) Multi-Class SVM>> ALGORITHM 1
def sub_problem_solve_search_l2(A,B,C,n_class,class_):
	D = B.copy()
	D[0],D[class_] = D[class_],D[0]
	sorted_indices = np.argsort(-D[1:])
	D[1:] = D[1:][sorted_indices]
	A_y = A+1/(2*C)

	beta_den,beta_num = A/A_y,D[0]*A/A_y
	beta = beta_num/beta_den
	r = 1
	while r<n_class and beta<D[r]:
		beta_num += D[r]
		beta_den += 1
		beta = beta_num/beta_den
		r+=1
	
	alpha_new  = np.clip((beta-B)/A,None,0)
	alpha_new[class_] = (beta-B[class_])/A_y
	
	return alpha_new

def solve_sub_problem(n_class,C,svm_l1_loss,sample_index,class_,gradient_sample,k,alpha_sample,unbounded_sample,compute_gradient,gradient):
	A,B = k[sample_index],gradient_sample-k[sample_index]*alpha_sample
	B[class_] -= 0 if svm_l1_loss else alpha_sample[class_]/(2*C)

	if svm_l1_loss:
		alpha_new = sub_problem_solve_search_l1(A,B,C,n_class,class_)
	else:
		alpha_new = sub_problem_solve_search_l2(A,B,C,n_class,class_)

	#更新gradient
	if compute_gradient:
		gradient += k[:,np.newaxis]*(alpha_new-alpha_sample)
		gradient_sample[class_] += 0 if svm_l1_loss else (alpha_new[class_]-alpha_sample[class_])/(2*C)
		unbounded_sample[:] = alpha_new<0
		unbounded_sample[class_] = alpha_new[class_]<C if svm_l1_loss else True
	alpha_sample[:] = alpha_new

def choose_sample_index(X,y,indices,active_unactive_set,n_class,C,individual_C,svm_l1_loss,alpha,kernel,kernel_args,kkt_tol,cooling_kkt_tol,compute_gradient,maintaining_active_set,K,gradient,unbounded):
	max_violate,max_violate_index,max_violate_class,max_violate_gradient,max_violate_k,max_violate_alpha,max_violate_unbounded = -np.inf,None,None,None,None,None,None
	turn,optimality = 0,False

	while turn<2: 
		if maintaining_active_set:
			sample_indices = indices[active_unactive_set[turn]]
		else:
			sample_indices = range(X.shape[0])
		
		for sample_index in sample_indices:
			C_sample = C[sample_index] if individual_C else C
			k = kernel(X,X[sample_index],*kernel_args) if K is None else K[sample_index] 
			class_ = y[sample_index]
			alpha_sample = alpha[sample_index]

			if compute_gradient:
				unbounded_sample = unbounded[sample_index]
				gradient_sample = gradient[sample_index]

			else:
				unbounded_sample = alpha_sample < 0
				unbounded_sample[class_] = alpha_sample[class_] < C_sample if svm_l1_loss else True
				
				#计算梯度。
				#当loss为Hinge-Loss时，梯度的计算见论文<<A Comparison of Methods for Multi-class Support Vector Machines>> IV节
				#当loss为Squared Hinge-Loss时，梯度的计算见论文<<A Study on L2-Loss (Squared Hinge-Loss) Multi-Class SVM>> 5.2节
				gradient_sample = np.dot(k,alpha) + 1
				gradient_sample[class_] += -1 + ( 0 if svm_l1_loss else alpha_sample[class_]/(2*C_sample) )
			
			violate = np.max(gradient_sample) - np.min(gradient_sample[unbounded_sample]) if np.any(unbounded_sample) else 0

			if violate > max_violate:
				max_violate,max_violate_index,max_violate_class = violate,sample_index,class_
				max_violate_gradient,max_violate_k = gradient_sample,k
				max_violate_alpha,max_violate_unbounded = alpha_sample,unbounded_sample
		
		if less_equal(max_violate,cooling_kkt_tol):
			turn += 1 if maintaining_active_set else 2	
		else:
			break
	
	if turn==2 and less_equal(max_violate,kkt_tol):
		optimality = True
	elif maintaining_active_set:
		active_unactive_set[0,max_violate_index] = True
		active_unactive_set[1,max_violate_index] = False

	return optimality,max_violate,max_violate_index,max_violate_class,max_violate_gradient,max_violate_k,max_violate_alpha,max_violate_unbounded

_cooling_scheme_linear = lambda init_tol,iter_index:init_tol/(iter_index+1)
_cooling_scheme_exp = lambda init_tol,iter_index:init_tol*exp(-iter_index)
_cooling_scheme_log = lambda init_tol,iter_index:init_tol/log10(iter_index+10)

def multisvm(X,y,alpha,sample_weights,C,kernel,kernel_args,svm_l1_loss,kkt_tol,max_iter,K,buf_ratio,n_class,maintaining_active_set=False,cooling=False,compute_gradient=False,init_cooling_kkt_tol=0.999,cooling_scheme='linear'):
	"""
		Crammer and Singer's multi-class svm。具体见论文<<A Comparison of Methods for Multi-class Support Vector Machines>>、<<A Study on L2-Loss (Squared Hinge-Loss) Multi-Class SVM>>、<<On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines>>
		参数：
			①n_class：int，表示类的数量
			②maintaining_active_set,cooling：bool，分别表示是否维持一个active set和使用cooling来提高算法效率。默认为False。具体作用见论文<<On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines>>第7节
			③compute_gradient：bool，表示是否计算并存储梯度。计算并存储梯度可以提高算法效率，但需要更多的存储空间。默认为False
			④init_cooling_kkt_tol：浮点数，当cooling为True时有效。具体作用见论文
			⑤cooling_scheme：str，当cooling为True时有效。其可选为'linear'、'exp'和'log'。默认为'linear'。具体作用见论文
			⑥cooling_turn_to_linear_niter_threshold：浮点数，当cooling为True时有效，表示将cooling_scheme置为'linear'的迭代阈值。当迭代步数t超过max_iter*cooling_turn_to_linear_niter_threshold时，cooling_scheme被置为'linear'。默认为1.0
	"""
	N = X.shape[0]
	
	if not sample_weights is None:
		C = sample_weights*C
		individual_C = True
	else:
		individual_C = False
	
	indices = np.arange(N,dtype=np.uint32)
	if compute_gradient:
		if alpha is None:
			unbounded = np.zeros((N,n_class),np.bool)
			unbounded[indices,y] = True
			gradient = np.ones((N,n_class),np.float64)
			gradient[indices,y] = 0
		else:
			unbounded = alpha<0
			unbounded[indices,y] = alpha[indices,y]<C if svm_l1_loss else True
			if not K is None:
				gradient = np.dot(K,alpha) + 1
			else:
				buf_size = min(N,max(1,int(N*buf_ratio)))
				gradient = np.empty((N,n_class),np.float64)
				start_index,end_index = 0,buf_size
				
				while start_index<N:
					K_ = kernel(X[start_index:end_index],X,*kernel_args)
					gradient[start_index:end_index] = np.dot(K_,alpha) + 1
					start_index,end_index = end_index,end_index+buf_size

			gradient[indices,y] += -1 + ( 0 if svm_l1_loss else alpha[indices,y]/(2*C) )
	else:
		unbounded,gradient = None,None
	
	if alpha is None:
		alpha = np.zeros((N,n_class),np.float64)

	if maintaining_active_set:
		active_unactive_set = np.empty((2,N),np.bool8)
		active_unactive_set[0],active_unactive_set[1] = False,True
	else:
		indices = None
		active_unactive_set = None
		cooling = False

	if cooling:
		cooling_scheme = _cooling_scheme_linear if cooling_scheme == 'linear' else\
						_cooling_scheme_exp if cooling_scheme=='exp' else \
						_cooling_scheme_log if cooling_scheme=='log' else \
						None
		if cooling_scheme is None:
			raise ValueError("cooling scheme must be linear','exp' or 'log'")
	else:
		cooling_kkt_tol = kkt_tol

	for iter_index in range(max_iter):
		if cooling:
			cooling_kkt_tol = cooling_scheme(init_cooling_kkt_tol,iter_index)
			if less_equal(cooling_kkt_tol,kkt_tol):
				cooling_kkt_tol = kkt_tol
				cooling = False

		optimality,max_violate,sample_index,class_,gradient_sample,k,alpha_sample,unbounded_sample = choose_sample_index(X,y,indices,active_unactive_set,n_class,C,individual_C,svm_l1_loss,alpha,kernel,kernel_args,kkt_tol,cooling_kkt_tol,compute_gradient,maintaining_active_set,K,gradient,unbounded)
		if optimality:
			break

		C_sample = C[sample_index] if individual_C else C
		solve_sub_problem(n_class,C_sample,svm_l1_loss,sample_index,class_,gradient_sample,k,alpha_sample,unbounded_sample,compute_gradient,gradient)

	return alpha
