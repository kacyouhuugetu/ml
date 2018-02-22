from math import ceil
from itertools import product
from .base import to_tuple_safe,check_sequence
import numpy as np

def n_fold_shuffler(N,n_fold=5):
	sample_indices = np.zeros(N,dtype=np.bool8)
	data_indices = np.empty((n_fold,2,N),dtype=np.bool8)
	fold_size = ceil(N/n_fold)
	test_fold_start = 0
	for fold_index in range(n_fold):
		sample_indices[test_fold_start:test_fold_start+fold_size] = True
		data_indices[fold_index] = (np.logical_not(sample_indices),sample_indices)
		sample_indices[test_fold_start:test_fold_start+fold_size] = False
		test_fold_start += fold_size
	return data_indices

def bootstrap(X,y,train_method,aggregate_method,bootstrap_count,bootstrap_size=None,aggregate_step=False,train_method_args=(),aggregate_method_args=(),dtype_indices=np.uint32):
	N = X.shape[0]
	bootstrap_size = N if bootstrap_size is None else bootstrap_size
	sample_indices = np.arange(N,dtype=dtype_indices)
	train_res = np.empty(bootstrap_count,np.object)
	for bootstrap_index in range(bootstrap_count):
		bootstrap_sample_indices = np.random.choice(sample_indices,size=bootstrap_size)
		if y is None:
			train_res[bootstrap_index] = train_method(X[bootstrap_sample_indices],*train_method_args)
		else:
			train_rse[bootstrap_index] = train_method(X[bootstrap_sample_indices],y[bootstrap_sample_indices],*train_method_args)
	return aggregate_method(train_res)

#parameters为一 n_parameter x m array-like
#当parameters为 n_parameter x 3 array-like时，parameters的每一行代表某一参数的取值范围，每一行的第一列代表取值的最小值，每一行的第二列代表取值的最大值，每一行的第三列代表取值的数量
#当parameters为 n_parameter x m (m>3)array-like时，parameters的每一行代表某一参数的所有取值可能，程序将直接在这些参数上进行
def cross_validation(X,y,sample_weights,n_fold,train_method,predict_method,loss_function,parameters=None,train_method_args=(),predict_method_args=(),loss_function_args=(),dtype_data=np.float64):
	if not parameters is None and not check_sequence(parameters):
		raise ValueError('parameters must be n_parameter x m array-like')
	
	if n_fold<0 or n_fold>X.shape[0]:
		raise ValueError('n_fold must be positive and less than N,where N is the length of X')
	
	if not parameters is None:
		try:
			parameters = [np.linspace(seq[0],seq[1],seq[2],dtype=np.float64) if len(seq)==3 else seq for seq in parameters]
		except:
			raise ValueError('parameters must be n_parameter x m array-like')
	
	data_slices = n_fold_shuffler(X.shape[0],n_fold)
	min_loss,min_parameters_value = np.inf,None

	iters = range(1) if parameters is None else product(*parameters)
	for parameters_value in iters:
		loss = 0
		for train_sample_indices,test_sample_indices in data_slices:
			sample_weights_train = None if sample_weights is None else sample_weights[train_sample_indices]
			sample_weights_test = None if sample_weights is None else sample_weights[test_sample_indices]
			train_args = to_tuple_safe(train_method_args) if parameters is None else parameters_value+to_tuple_safe(train_method_args)
			train_res = train_method(X[train_sample_indices],y[train_sample_indices],sample_weights_train,*train_args)

			predict_res = predict_method(X[test_sample_indices],*(to_tuple_safe(train_res)+to_tuple_safe(predict_method_args)))

			loss_args = to_tuple_safe(predict_res)+to_tuple_safe(loss_function_args) if parameters is None else to_tuple_safe(predict_res)+parameters_value+to_tuple_safe(loss_function_args)
			loss += loss_function(y[test_sample_indices],sample_weights_test,*loss_args)
		if loss<min_loss:
			min_loss,min_parameters_value = loss,parameters_value
	
	return min_parameters_value
