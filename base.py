from math import sqrt,pi,ceil
from bisect import bisect_left,bisect_right
from scipy.special import expit as logistic_fun
from scipy.special import erf
from sklearn.utils.extmath import log_logistic as log_logistic_fun
from .myfun import unique
import numpy as np

isclose = lambda a,b,tol=1e-8:True if a==b else abs(a-b)<=tol
less_equal = lambda a,b,tol=1e-8:a<b or isclose(a,b,tol)
large_equal = lambda a,b,tol=1e-8:a>b or isclose(a,b,tol)

#检查某一array是否具有指定的维数
def check_array(array,dim=2):
	if isinstance(array,np.ndarray) and (dim is None or len(array.shape)==dim):
			return True
	return False

def check_sequence(seq):
	if type(seq) in (tuple,list,np.ndarray):
		return True
	return False

#检查两个array是否具有各自指定的维数，且第一个维度的长度相同
def check_same_length(array_a,array_b,dim_a=2,dim_b=None):
	if check_array(array_a,dim_a) and check_array(array_b,dim_b) and array_a.shape[0]==array_b.shape[0]:
		return True
	return False

def to_tuple_safe(obj,ignore_None=True):
	if type(obj) in (tuple,list):
		return tuple(obj)
	elif obj is None and ignore_None:
		return ()
	else:
		return (obj,)

def data_normalize(data):
	data_means = np.mean(data,axis=0)
	data_stds = np.std(data,axis=0,ddof=1)
	normalized_data = (data-data_means)/data_stds
	return normalized_data,data_means,data_stds

def data_unnormalize(data,data_means,data_stds):
	return data*data_stds+data_means

identity = lambda x:x
ones = lambda x:np.ones_like(x)

#log(x!)的Stirling's approximation，其中x为正整数
def log_factorial(x):
	return x*np.log(x)-x

def neg_log_likelihood_poisson(y,mu,N=None,p=None,delta=1e-8):
	#np.clip避免当mu为0或负值
	return -np.sum(y*np.log(np.clip(mu,delta,None))-mu-log_factorial(y))

def neg_log_likelihood_logistic(y,mu,N=None,p=None,delta=1e-8):
	pos_class_indices,neg_class_indices = y==1,y!=1
	pos_probs,neg_probs = mu[pos_class_indices],1-mu[neg_class_indices]
	pos_probs[pos_probs<delta] = delta
	neg_probs[neg_probs<delta] = delta
	return -np.sum(np.log(pos_probs))-np.sum(np.log(neg_probs))

_loss_logistic_gradient_z = lambda y,z,w=None:logistic_fun(-y*z)-(y+1.)/2. if w is None else w*(logistic_fun(-y*z)-(y+1.)/2.)

reci_sqrt_2pi = 1./sqrt(2*pi)
standard_normal_pdf = lambda x:reci_sqrt_2pi*np.exp(-x**2/2)
standard_normal_cdf = lambda x,x_div_sqrt_2=None:(erf(x/sqrt(2)) + 1.)/2. if x_div_sqrt_2 is None else (erf(x_div_sqrt_2) + 1.)/2.

def linear_kernel(X1,X2,X2_T=None):
	X2_T = X2.T if X2_T is None else X2_T
	if X2_T.ndim==1:
		return X1.dot(X2_T)
	return X1.dot(X2_T)

def get_norm2_square(X1,X2,dtype=np.float64):
	if X2.ndim==1 or X1.ndim==1:
		norm2_square = np.sum((X1-X2)**2,axis=-1) if X2.ndim==1 else np.sum((X2-X1)**2,axis=1)
	else:
		norm2_square_matrix = np.empty((X1.shape[0],X2.shape[0]),dtype)
		for sample_index in range(X1.shape[0]):
			x = X1[sample_index]
			norm2_square = np.sum((X2-x)**2,axis=1)
			norm2_square_matrix[sample_index] = norm2_square
	
		return norm2_square_matrix

def gaussian_kernel(X1,X2,precision,norm2_square=None,return_derivative=False,dtype=np.float64):
	if X2.ndim==1 or X1.ndim==1:
		if norm2_square is None:
			norm2_square = np.sum((X1-X2)**2,axis=-1) if X2.ndim==1 else np.sum((X2-X1)**2,axis=1)
		kernel = np.exp(-precision*norm2_square)
		return (kernel,kernel*-norm2_square) if return_derivative else kernel

	if norm2_square is None:
		kernel_matrix = np.empty((X1.shape[0],X2.shape[0]),dtype)
		derivative = np.empty_like(kernel_matrix)
		for sample_index in range(X1.shape[0]):
			x = X1[sample_index]
			norm2_square = np.sum((X2-x)**2,axis=1)
			kernel_matrix[sample_index] = np.exp(-precision*norm2_square)
			if return_derivative:
				derivative[sample_index] = kernel_matrix[sample_index]*-norm2_square
	else:
		kernel_matrix = np.exp(-precision*norm2_square)
		if return_derivative:
			derivative = kernel_matrix*-norm2_square
	return (kernel_matrix,derivative) if return_derivative else kernel_matrix

def _classify_pre_process(X,y,compute_weights=True,sample_weights=None,sorted=False):
	N,p = X.shape
	if compute_weights and sample_weights is None:
		sample_weights = np.ones(N,X.dtype)
	if not sorted:
		sorted_indices = np.argsort(y)
		X,y = X[sorted_indices],y[sorted_indices]
		if compute_weights:
			sample_weights = sample_weights[sorted_indices]

	class_names,class_indices,class_counts = unique(y,return_index=True,return_counts=True,sorted=True)
	class_indices = np.append(class_indices,N)
	class_slices = np.array([slice(class_indices[index],class_indices[index+1]) for index in range(len(class_names))])
	if compute_weights:
		class_weights = np.array([np.sum(sample_weights[class_slice]) for class_slice in class_slices],dtype=sample_weights.dtype)
	else:
		class_weights = None
	return X,y,sample_weights,class_names,class_weights,class_slices,class_counts

def one_versus_rest_train(X,y,train_method,train_args=(),class_slices=None,sorted=False):
	if class_slices is None:
		X,y,_,_,_,class_slices,_ = _classify_pre_process(X,y,False,sorted=sorted)
	N,n_class = X.shape[0],len(class_slices)
	
	train_res = np.empty(n_class,np.object)
	class_slice,class_other_slice = np.zeros(N,np.bool8),np.ones(N,np.bool8)
	target = np.empty(N,np.int8)
	target[:] = -1
	for class_index in range(n_class):
		slice_ = class_slices[class_index]
		target[slice_] = 1
		train_res[class_index] = train_method(X,target,*train_args)
		target[slice_] = -1

	return train_res

def one_versus_one_train(X,y,train_method,train_args=(),class_slices=None,sorted=False):
	if class_slices is None:
		X,y,_,_,_,class_slices,_ = _classify_pre_process(X,y,False,sorted=sorted)
	N,n_class = X.shape[0],len(class_slices)

	train_res = np.empty(n_class*(n_class-1)//2,np.object)
	
	target = np.empty(N,np.int8)
	indices = np.zeros(N,np.bool8)
	index = 0
	for class_index in range(n_class):
		slice_ = class_slices[class_index]
		indices[slice_] = True 
		target[slice_] = 1
		for class_other_index in range(class_index+1,n_class):
			slice_other = class_slices[class_other_index]
			target[slice_other] = -1
			indices[slice_other] = True

			train_res[index] = train_method(X[indices],target[indices],*train_args)
			
			indices[slice_other] = False
			index += 1
		indices[slice_] = False

	return train_res
	
def one_versus_rest_predict(X,n_class,train_res,predict_method,predict_args=(),class_names=None):
	n_class = len(train_res)
	if class_names is None:
		class_names = np.arange(n_class,dtype=np.int16)
	predict_args = to_tuple_safe(predict_args)

	score = np.empty((n_class,X.shape[0]),np.float64)
	for class_index in range(n_class):
		train_res_ = train_res[class_index]
		predict_args_ = to_tuple_safe(train_res_) + predict_args
		score[class_index] = predict_method(X,*predict_args_)
		
	predict = class_names[np.argmax(score,axis=0)]
	return predict

def one_versus_one_predict(X,n_class,train_res,predict_method,predict_args=(),class_names=None):
	if class_names is None:
		class_names = np.arange(n_class,dtype=np.int16)
	predict_args = to_tuple_safe(predict_args)
	
	index = 0
	vote = np.zeros((n_class,X.shape[0]),np.float64)
	for class_index in range(n_class):
		for class_other_index in range(class_index+1,n_class):
			train_res_ = train_res[index]
			predict_args_ = to_tuple_safe(train_res_) + predict_args
			score = predict_method(X,*predict_args_)
			vote[class_index,score>0] += 1
			vote[class_other_index,score<0] += 1
			index+=1

	predict = class_names[np.argmax(vote,axis=0)]
	return predict


def sort_union(l1,l2,inplace=False):
	if not inplace:
		l1 = l1.copy()
	
	if len(l1)==0:
		l1.extend(l2)
		return l1

	len_l2,index,l2_index = len(l2),0,0
	end_ele = l1[-1]
	while l2_index<len_l2 and l2[l2_index]<=end_ele:
		if l1[index]<l2[l2_index]:
			index += 1
		
		else:
			if l1[index]>l2[l2_index]:
				l1.insert(index,l2[l2_index])
			index += 1
			l2_index += 1

	if l2_index<len_l2:
		l1.extend(l2[l2_index+(1 if end_ele==l2[l2_index] else 0):])
	
	return l1

def sort_difference(l1,l2,inplace=False):
	if not inplace:
		l1 = l1.copy()

	if len(l1)==0:
		return l1

	len_l2,index,l2_index = len(l2),0,0
	end_ele = l1[-1]
	while l2_index<len_l2 and l2[l2_index]<=end_ele:
		if l1[index]<l2[l2_index]:
			index += 1
		elif l1[index]>l2[l2_index]:
			l2_index += 1
		else:
			l1.pop(index)
			l2_index += 1
	return l1

def sort_intersection(l1,l2,inplace=False):
	if not inplace:
		l1 = l1.copy()

	len_l1,len_l2,index,l2_index = len(l1),len(l2),0,0
	if len_l1==0 or len_l2==0:
		return []

	end_ele = l1[-1]
	while index<len_l1 and l2_index<len_l2:
		if l1[index]<l2[l2_index]:
			len_l1 -= 1
			l1.pop(index)
		elif l1[index]>l2[l2_index]:
			l2_index += 1
		else:
			index += 1
			l2_index += 1
	if len_l1>index and l2[-1]<l1[index]:
		for _ in range(len_l1-index):
			l1.pop(index)

	return l1

def sort_search(l,ele):
	find_index = bisect_left(l,ele)
	if find_index==len(l) or l[find_index]!=ele:
		return None
	else:
		return find_index

def sort_search_continuous_left_most(l,ele_index,min_dist=5,left=None):
	right,left = ele_index,0 if left is None or left<0 or left>ele_index else left
	ele = l[right]
	while True:
		if right-left+1<min_dist:
			if left==right:
				return left
			ele -= right
			index,early_stop = right,False
			for index in range(right-1,left-1,-1):
				if l[index]!=ele+index:
					early_stop=True
					break
			return index+early_stop

		mid = (left+right)//2
		if l[mid]==ele+mid-right:
			right = mid
			ele = l[right]
		else:
			left = mid+1

def sort_search_continuous_right_most(l,ele_index,min_dist=5,right=None):
	left,right = ele_index,len(l)-1 if right is None or right>=len(l) else right
	ele = l[left]
	while True:
		if right-left+1<min_dist:
			if left==right:
				return left
			ele -= left
			early_stop = False
			for index in range(left+1,right+1):
				if l[index]!=ele+index:
					early_stop=True
					break
			return index-early_stop

		mid = (left+right)//2
		if l[mid]==ele+mid-left:
			left = mid
			ele = l[left]
		else:
			right = mid-1

#find minimal index of ele which is larger equal than min_ele
#return len(l) if can not find such a index
def sort_search_left_most(l,min_ele):
	find_index = bisect_right(l,min_ele)
	return find_index - (find_index>0 and l[find_index-1]==min_ele)

#find maximal index of ele which is less equal than max_ele
#return -1 if can not find such a index
def sort_search_right_most(l,max_ele):
	find_index = bisect_left(l,max_ele)
	return find_index - (find_index==len(l) or l[find_index]!=max_ele)

def partition_data_space(X,n_intervals=None,intervals_size=None):
	p = X.shape[1]
	if not n_intervals is None:
		if not type(n_intervals) in (tuple,list,np.ndarray):
			n_intervals = (n_intervals,)*p
		ranges = (None,)*p

	elif not intervals_size is None:
		if not type(intervals_size) in (tuple,list,np.ndarray):
			intervals_size = (intervals_size,)*p
		n_intervals,ranges = [],[]
		for attribute_index in range(p):
			a_min,a_max,interval_size = np.min(X[:,attribute_index]),np.max(X[:,attribute_index]),intervals_size[attribute_index]
			
			n_interval = ceil((a_max-a_min)/interval_size)
			extra_size = n_interval*interval_size-(a_max-a_min)
			a_min,a_max = a_min-extra_size/2,a_max+extra_size/2
			
			n_intervals.append(n_interval)
			ranges.append((a_min,a_max))

	else:
		raise ValueError("error:n_intervals and intervals_size are both None")

	histogram,codes = np.empty(p,np.object),np.empty((X.shape[0],p),np.uint32)
	knots_data,knots_codes = np.empty(p,np.object),np.empty(p,np.uint32)
	
	for attribute_index in range(p):
		x,n_interval = X[:,attribute_index],n_intervals[attribute_index]
		histogram[attribute_index],knots = np.histogram(x,bins=n_interval,range=ranges[attribute_index])
		knots = knots[:-1]	
		code,cumsum = np.searchsorted(knots,x,side='right')-1,np.sum(n_intervals[:attribute_index])
		codes[:,attribute_index] = code + cumsum

		knots_data[attribute_index] = knots
		knots_codes[attribute_index] = cumsum

	return knots_data,knots_codes,codes,histogram
