#multi-layer neural network，已实现普通神经网络和卷积神经网络
import numpy as np
from scipy.optimize import minimize
from math import log
from time import clock
import sys
import codecs

def predict(nn,X):
	if X.ndim<2:
		X = X[np.newaxis,:]
	N = X.shape[0]
	outputs = np.empty((N,nn._n_units_output),np.float)
	for index in range(N):
		outputs[index],_ = nn._forward_propagate(nn._weights,X[index])
	return outputs

def loss(weights,nn,read_row):
	return nn.network_gradient(weights,read_row)

def jac(weights,nn,read_row):
	return nn._gradient

def hessp(weights,V,nn,read_max):
	dtype_data = nn._dtype_data
	weights_derivate,outputs_second_derivate = nn._weights_derivate,nn._outputs_second_derivate
	n_weights = nn._n_weights
	store_file = nn._store_file
	hess_product = np.zeros(n_weights,dtype_data)
	weights_derivate_shape = weights_derivate.shape
	read_count = nn._read_count
	read_amount = np.prod(weights_derivate_shape)

	product = np.dot(weights_derivate[:read_count],V)
	product = np.einsum('ijk,ik->ij',outputs_second_derivate[-read_count:],product)
	hess_product += np.einsum('ijk,ij->k',weights_derivate[-read_count:],product)
	
	store_file.seek(0)
	read_index = 0
	for read_block_index in range((nn._X.shape[0]-1)//read_max):
		weights_derivate[:] = np.reshape(np.fromfile(store_file,dtype_data,read_amount),weights_derivate_shape)
		product = np.dot(weights_derivate,V)
		product = np.einsum('ijk,ik->ij',outputs_second_derivate[read_index:read_index+read_max],product)
		hess_product += np.einsum('ijk,ij->k',weights_derivate,product)
		read_index+=read_max

	return hess_product
			

class fully_connected_network:
	def __init__(self,n_units_input,n_units,activation_fun,activation_derivate_fun,dtype_data=np.float32,dtype_indices=np.uint16):
		self._n_layers = len(n_units)
		n_units = np.array([n_units_input]+list(n_units),dtype_indices)
		self._weights_shapes = np.hstack((n_units[1:,np.newaxis],n_units[:-1,np.newaxis]+1))
		self._weights_stops = np.cumsum(np.prod(self._weights_shapes,1,dtype_indices))
		self._network_activations = np.array([np.empty(n_unit,dtype_data) for n_unit in n_units[1:]],np.object)
		self._activation_fun,self._activation_derivate_fun = activation_fun,activation_derivate_fun
		self._n_units_network_output = n_units[-1]
		self._n_weights = self._weights_stops[-1]

	def _forward_propagate(self,weights,activations_down):
		activation_fun = self._activation_fun
		network_activations = self._network_activations
		weights_shapes,weights_stops = self._weights_shapes,self._weights_stops
		weights_start = 0
		for layer_index in range(self._n_layers):
			weights_stop = weights_stops[layer_index]
			W = np.reshape(weights[weights_start:weights_stop],weights_shapes[layer_index])
			nets = np.dot(W[:,1:],activations_down)+W[:,0]
			activations_down = activation_fun(nets)
			network_activations[layer_index] = activations_down
			weights_start = weights_stop
		
		return activations_down

	def _backward_propagate(self,n_units_output,weights,weights_derivate,W_up,beta_cum_up,activations_down_other_network):
		activation_derivate_fun = self._activation_derivate_fun
		network_activations = self._network_activations
		weights_shapes,weights_stops = self._weights_shapes,self._weights_stops
		weights_stop = weights_stops[-1]
		activations_current = network_activations[-1]
		for layer_index in range(self._n_layers-1,-1,-1):
			weights_start = weights_stops[layer_index-1] if layer_index>0 else 0
			activations_down = network_activations[layer_index-1] if layer_index>0 else activations_down_other_network
			beta_up = W_up[:,1:]*activation_derivate_fun(activations_current)
			beta_cum_up = np.dot(beta_cum_up,beta_up)
			weights_derivate[:,weights_start:weights_stop] = np.reshape(np.hstack((np.reshape(beta_cum_up,(-1,1)),np.outer(beta_cum_up,activations_down))),(n_units_output,weights_stop-weights_start))
			W_up = np.reshape(weights[weights_start:weights_stop],weights_shapes[layer_index])
			activations_current = activations_down
			weights_stop = weights_start
		
		return beta_cum_up,W_up

class neural_network:
	def __init__(self,task,n_units_input,n_units_output,networks_types,networks_args,dtype_data=np.float32,dtype_indices=np.uint16):
		if len(networks_types) != len(networks_args):
			raise ValueError("networks_type don't match networks_args:incosistent size")
		
		n_networks = len(networks_types)
		networks = np.empty(n_networks,np.object)
		weights_stops = np.empty(n_networks,dtype_indices)
		n_units_output_pre_network = n_units_input
		weights_stops_pre_network = 0
		for network_index in range(n_networks):
			network_type,network_args = networks_types[network_index],networks_args[network_index]
			#......检查network_args为序列.......
			
			#fully connected network
			#network_args是长度为3的序列
			#第一个元素为包含各层hidden layer的hiddent units数目的序列
			#第二个元素为各个units所用的激活函数,第三个元素是激活函数的导数
			if network_type == 'F':
				network = fully_connected_network(n_units_output_pre_network,*network_args,dtype_data=dtype_data,dtype_indices=dtype_indices)
			#convolution network
			elif network_type == 'C':
				network = convolution_network(n_units_output_pre_network,*network_args,dtype_data=dtype_data,dtype_indices=dtype_indices)
			else:
				raise ValueError('Unknown network type:%s'%layer)
			
			networks[network_index] = network
			weights_stops[network_index] = weights_stops_pre_network + network._n_weights
			n_units_output_pre_network = network._n_units_network_output
			weights_stops_pre_network = weights_stops[network_index]
		output_weights_shape = (n_units_output,n_units_output_pre_network+1)
		
		self._dtype_data = dtype_data
		self._n_units_input,self._n_units_output,self._n_networks = n_units_input,n_units_output,n_networks
		self._networks = networks
		self._weights_stops = weights_stops
		self._output_weights_shape = output_weights_shape
		
		n_weights = self._weights_stops[-1]+output_weights_shape[0]*output_weights_shape[1]
		self._n_weights = n_weights
		self._weights,self._gradient = np.empty(n_weights,dtype_data),np.empty(n_weights,dtype_data)

		if task=='regression':
			self._output_fun = lambda z:z
			self._output_derivate_fun = lambda z,z_:z-z_
			self._output_second_derivate_fun = lambda z:np.eye(n_units_output)
			self._output_error_fun = lambda z,z_:np.sum((z-z_)**2)
		else:
			def output_derivate_fun(z,z_):
				z = z.copy()
				z[z_] = z[z_] - 1
				return z
			self._output_fun = lambda z:np.exp(z)/np.sum(np.exp(z))
			self._output_derivate_fun = output_derivate_fun
			self._output_second_derivate_fun = lambda z:np.diag(z)-np.outer(z,z)
			self._output_error_fun = lambda z,z_:-log(z[z_])

	def train(self,X,Y,init_weights=None,weights_vars=0.1,weights_max=None,store_file_name='_stored_derivate_data'):
		self._X,self._Y = X,Y
		N = X.shape[0]
		n_weights,n_units_output = self._n_weights,self._n_units_output
		dtype_data = self._dtype_data
		if init_weights is None:
			init_weights = np.random.normal(0,weights_vars,n_weights)
		if not weights_max is None:
			init_weights[init_weights>weights_max] = weights_max
		if sys.stdin.encoding != sys.stdout.encoding:
			sys.stdin = codecs.getreader(sys.stdout.encoding)(sys.stdin)
		nbytes = N*n_units_output*n_weights*np.dtype(dtype_data).itemsize
		nGB = nbytes//1e9
		nMB = (nbytes//1e6)%1e6
		nKB = (nbytes//1e3)%1e3
		print('存储整个weights_derivate数据所需的内存大小为:%03d,%03d,%03d KB\n选择存储于主存中的数据的比例(1表示完全存储):'%(nGB,nMB,nKB),end='')
		read_row_rate = float(input())
		read_max = min(n_weights,max(1,int(N*read_row_rate)))
		self._store_file = open(store_file_name,'w+b')
		self._weights_derivate = np.empty((read_max,n_units_output,n_weights),dtype_data)
		self._outputs_second_derivate = np.empty((N,n_units_output,n_units_output),dtype_data)
		
		res = minimize(loss,init_weights,args=(self,read_max),jac=jac,hessp=hessp,method='Newton-CG')
		self._store_file.close()
		return res

	def _forward_propagate(self,weights,x):
		n_networks = self._n_networks
		networks = self._networks
		weights_stops = self._weights_stops
		output_weights_shape = self._output_weights_shape
		output_fun = self._output_fun

		network_activations = [x]
		activations_down = x
		weights_start = 0
		for network_index in range(n_networks):
			weights_stop = weights_stops[network_index]
			activations_down = networks[network_index]._forward_propagate(weights[weights_start:weights_stop],activations_down)
			network_activations.append(activations_down)
			weights_start = weights_stop
		
		W_up = np.reshape(weights[weights_start:],output_weights_shape)
		outputs = output_fun(np.dot(W_up[:,1:],activations_down)+W_up[:,0])

		return outputs,network_activations
	
	def _backward_propagate(self,weights,weights_derivate,x,network_activations):
		n_units_output = self._n_units_output
		n_networks = self._n_networks
		networks = self._networks
		weights_stops = self._weights_stops
		output_weights_shape = self._output_weights_shape
		eye_output = np.eye(n_units_output,dtype=self._dtype_data)
		
		activations_down = network_activations.pop()
		weights_start = weights_stop = weights_stops[-1]
		beta_cum_up = 1
		W_up = np.reshape(weights[weights_start:],output_weights_shape)
		weights_derivate[:,weights_start:] = np.reshape(np.hstack((np.reshape(eye_output,(-1,1)),np.outer(eye_output,activations_down))),(n_units_output,weights_derivate.shape[-1]-weights_start))
		
		for network_index in range(n_networks-1,-1,-1):
			activations_down = network_activations.pop()
			weights_start = weights_stops[network_index-1] if network_index>0 else 0
			beta_cum_up,W_up = networks[network_index]._backward_propagate(n_units_output,weights[weights_start:weights_stop],weights_derivate[:,weights_start:weights_stop],W_up,beta_cum_up,activations_down)
			weights_stop = weights_start
	
	def network_gradient(self,weights,read_max):
		t_start = clock()
		self._weights[:] = weights
		
		n_units_output = self._n_units_output
		n_weights = self._n_weights
		X,Y = self._X,self._Y
		gradient = self._gradient
		dtype_data = self._dtype_data
		output_derivate_fun = self._output_derivate_fun
		output_second_derivate_fun = self._output_second_derivate_fun
		output_error_fun = self._output_error_fun
		weights_derivate,outputs_second_derivate = self._weights_derivate,self._outputs_second_derivate
		gradient[:] = 0
		N = X.shape[0]
		store_file = self._store_file
		
		error = 0
		correct = 0
		
		store_file.seek(0)
		read_count = 0
		for sample_index in range(N):
			if read_count == read_max:
				weights_derivate.tofile(store_file)
				read_count = 0
			weights_derivate_ = weights_derivate[read_count]
			outputs_second_derivate_ = outputs_second_derivate[sample_index]
			
			x,y = X[sample_index],Y[sample_index]
			outputs,network_activations = self._forward_propagate(weights,x)
			self._backward_propagate(weights,weights_derivate_,x,network_activations)
			
			error += output_error_fun(outputs,y)
			gradient+=np.dot(weights_derivate_.T,output_derivate_fun(outputs,y))
			outputs_second_derivate_[:] = output_second_derivate_fun(outputs)

			read_count += 1


			if np.argmax(outputs)==y:
				correct += 1

		self._read_count = read_count
		use_time = int(clock()-t_start)
		use_time_second,use_time_minite,use_time_hour = use_time//3600,(use_time//60)%60,use_time%60
		print('-'*40)
		print('error:%f'%error)
		print('correct_rate:%f'%(correct/X.shape[0]))
		print('use_time:%02d:%02d:%02d'%(use_time_second,use_time_minite,use_time_hour))
		return error

#conv层units的net计算方法：
#假设低一层的activations为x，其为shape为(plane_length**2,N)1D array
#w为shape为(window_length**2+1,N)的2D array，即w[:,i]为第i层conv层的权重向量
#activations_down = x[gen_index(plane_length,window_length),:]
#则net = np.sum(activations_down*w[1:],axis=1)+w[0]

#若conv_layer=True，则表示生成conv层的forward_indices，plane_length表示低一层的ss层的长度
#返回该层的长度和forward_indices
def gen_index(plane_length,window_length,conv_layer,dtype_indices):
	L = plane_length-window_length+1 if conv_layer else plane_length//window_length
	indices = np.zeros((L,L,window_length,window_length),dtype_indices)
		
	indices[0,0] = np.reshape(np.arange(window_length**2,dtype=dtype_indices),(window_length,window_length))+np.array([i*(plane_length-window_length) for i in range(window_length)],dtype_indices)[:,np.newaxis]
	
	row_offset = plane_length if conv_layer else plane_length*window_length
	col_offset = 1 if conv_layer else window_length 
	for row_index in range(1,L):
			indices[row_index,0] = indices[row_index-1,0]+row_offset
	for col_index in range(1,L):
			indices[:,col_index] = indices[:,col_index-1]+col_offset
	return L,np.reshape(indices,(L**2,window_length**2))

#sub-sample层beta_cum的计算方法：(结果为shape为(n_units_output,plane_length**2)的2D array)
#首先对高一层的beta_cum_up进行左补零，即beta_cum_up_ = np.hstack((np.zeros((n_units_output,1)),beta_cum_up))
#indices = gen_inv_index(plane_length,window_length)
#则beta_cum = np.dot(beta_cum_up_[:,indices],w[1:])

#若有N层不同权重的sub-sample层，各个beta_cum的计算方法为
#(高一层的beta_cum_up为(n_units_output,(plane_length-window_length+1)**2,N)的3D array,beta_cum结果为shape为(n_units_output,plane_length**2,N)的3D array)
#令W为shape为(window_length**2+1,N)的权重矩阵，即W[:,i]为第i层sub-sample的权重向量
#beta_cum = np.sum(beta_cum_up_[:,indices,:]*W[1:],axis=2)
#plane_length为ss层的plane_length
def gen_inv_index_ss(plane_length,window_length,dtype_indices):
	L = plane_length-window_length+1
	indices = np.zeros((plane_length,plane_length,window_length,window_length),dtype_indices)
		
	indices_ = np.reshape(np.arange(1,L**2+1,dtype=dtype_indices),(L,L))
	for col_index in range(window_length):
			indices[:L,col_index:col_index+L,0,col_index] = indices_
		
	indices_ = indices[:L,:,0]
	for row_index in range(1,window_length):
			indices[row_index:row_index+L,:,row_index] = indices_

	return np.reshape(indices,(plane_length**2,window_length**2))

#plane_length为conv层的plane_length
def gen_inv_index_conv(plane_length,window_length,dtype_indices):
	L = plane_length//window_length
	indices = np.reshape(np.repeat(np.arange(L**2,dtype=dtype_indices),window_length),(L,L*window_length))
	return np.repeat(indices,window_length,axis=0).flatten()

class convolution_network:
	def __init__(self,n_units_down,n_layers,n_planes,conv_window_length,ss_window_length,activation_fun,activation_derivate_fun,dtype_data=np.float32,dtype_indices=np.uint16):
		conv_window_size,ss_window_size = conv_window_length**2,ss_window_length**2
		forward_indices = np.empty((n_layers,2),dtype=np.object)
		backward_indices = np.empty((n_layers,2),dtype=np.object)
		network_activations = np.empty((n_layers,2),dtype=np.object)
		weights_stops = np.empty((n_layers,2),dtype_indices)
		weights_stops[:] = np.reshape(np.cumsum(np.repeat((((conv_window_size+1)*n_planes,2*n_planes),),n_layers,axis=0)),(n_layers,2))
		conv_non_paded_indices = np.empty(n_layers,dtype=np.object)
		plane_length = int(np.sqrt(n_units_down))
		for layer_index in range(n_layers):
			plane_length,forward_indices[layer_index,0] = gen_index(plane_length,conv_window_length,True,dtype_indices)
			needed_pad = plane_length%ss_window_length
			needed_pad = ss_window_length - needed_pad if needed_pad>0 else 0
			plane_paded_length = plane_length + needed_pad
			left_border,right_border = needed_pad>>1,plane_length+(needed_pad>>1)
			conv_non_paded_indices[layer_index] = np.reshape(np.arange(plane_paded_length**2,dtype=dtype_indices),(plane_paded_length,plane_paded_length))[left_border:right_border,left_border:right_border].flatten()
			network_activations[layer_index,0] = np.zeros(((plane_paded_length)**2,n_planes),dtype_data)
			backward_indices[layer_index,0] = gen_inv_index_conv(plane_paded_length,ss_window_length,dtype_indices)
			
			plane_length,forward_indices[layer_index,1] = gen_index(plane_paded_length,ss_window_length,False,dtype_indices)
			network_activations[layer_index,1] = np.empty((plane_length**2,n_planes),dtype_data)
			backward_indices[layer_index,1] = gen_inv_index_ss(plane_length,conv_window_length,dtype_indices) if layer_index < n_layers-1 else None
		
		self._n_layers,self._n_planes = n_layers,n_planes
		self._n_units_network_output_per_planes = plane_length**2
		self._n_units_network_output = n_planes*plane_length**2
		self._n_weights = weights_stops[-1,1]
		self._conv_non_paded_indices = conv_non_paded_indices
		self._network_activations = network_activations
		self._forward_indices,self._backward_indices = forward_indices,backward_indices
		self._conv_window_length = conv_window_length
		self._ss_window_length = ss_window_length
		self._weights_stops = weights_stops
		self._activation_fun,self._activation_derivate_fun = activation_fun,activation_derivate_fun

	def _forward_propagate(self,weights,activations_down):
		n_layers,n_planes = self._n_layers,self._n_planes
		conv_non_paded_indices = self._conv_non_paded_indices
		forward_indices = self._forward_indices
		network_activations = self._network_activations
		conv_window_length,ss_window_length = self._conv_window_length,self._ss_window_length
		conv_window_size,ss_window_size = conv_window_length**2,ss_window_length**2
		activation_fun = self._activation_fun
		weights_stops = self._weights_stops
		weights_start = 0

		activations_down = activations_down[:,np.newaxis]
		for layer_index in range(n_layers):
			conv_non_paded_index = conv_non_paded_indices[layer_index]
			weights_stop = weights_stops[layer_index,0]
			W = np.reshape(weights[weights_start:weights_stop],(conv_window_size+1,n_planes))
			nets = np.sum(activations_down[forward_indices[layer_index,0],:]*W[1:],axis=1)+W[0]
			activations_down = activation_fun(nets)
			network_activations[layer_index,0][conv_non_paded_index] = activations_down
			activations_down = network_activations[layer_index,0]
			weights_start = weights_stop

			weights_stop = weights_stops[layer_index,1]
			W = np.reshape(weights[weights_start:weights_stop],(2,n_planes))
			nets = np.sum(activations_down[forward_indices[layer_index,1],:],axis=1)*(W[1]/ss_window_size)+W[0]
			activations_down = activation_fun(nets)
			network_activations[layer_index,1] = activations_down
			weights_start = weights_stop

		return activations_down.flatten()

	def _backward_propagate(self,n_units_output,weights,weights_derivate,W_up,beta_cum_up,activations_down_other_network):
		n_layers,n_planes = self._n_layers,self._n_planes
		conv_non_paded_indices = self._conv_non_paded_indices
		forward_indices,backward_indices = self._forward_indices,self._backward_indices
		conv_window_length,ss_window_length = self._conv_window_length,self._ss_window_length
		conv_window_size,ss_window_size = conv_window_length**2,ss_window_length**2
		weights_stops = self._weights_stops
		network_activations = self._network_activations
		activation_derivate_fun = self._activation_derivate_fun
		activations_down_other_network = activations_down_other_network[:,np.newaxis]
		
		activations_down,activations_current = network_activations[-1]
		activations_derivate = activation_derivate_fun(activations_current)
		beta_up = W_up[:,1:]*activations_derivate.flatten()
		beta_cum_up = np.reshape(np.dot(beta_cum_up,beta_up),(n_units_output,self._n_units_network_output_per_planes,n_planes))
		weights_start,weights_stop = weights_stops[-1,0],weights_stops[-1,1]
		indices = forward_indices[-1,1]
		weights_derivate[:,weights_start:weights_stop] = np.hstack((np.sum(beta_cum_up,axis=1),np.sum(beta_cum_up*np.sum(activations_down[indices],axis=1)/ss_window_size,axis=1)))
		W_up = np.reshape(weights[weights_start:weights_stop],(2,n_planes))
		
		activations_down,activations_current = network_activations[-2,1] if n_planes>1 else activations_down_other_network,activations_down
		
		conv_non_paded_index = conv_non_paded_indices[-1]
		activations_derivate = activation_derivate_fun(activations_current[conv_non_paded_index])
		beta_cum_up = beta_cum_up[:,backward_indices[-1,0][conv_non_paded_index],:]*activations_derivate*(W_up[1]/ss_window_size)
		weights_start,weights_stop = weights_stops[-2,1] if n_layers>1 else 0,weights_start
		indices = forward_indices[-1,0]
		weights_derivate[:,weights_start:weights_stop] = np.hstack((np.sum(beta_cum_up,axis=1),np.reshape(np.einsum('ijk,jlk->ilk',beta_cum_up,activations_down[indices,:]),(n_units_output,conv_window_size*n_planes))))
		W_up = np.reshape(weights[weights_start:weights_stop],(conv_window_size+1,n_planes))
		activations_current = activations_down

		zeros = np.zeros((n_units_output,1,n_planes),np.float)
		for layer_index in range(n_layers-2,-1,-1):
			conv_non_paded_index = conv_non_paded_indices[layer_index]
			activations_down = network_activations[layer_index,0]
			activations_derivate = activation_derivate_fun(activations_current)
			beta_cum_up_ = np.concatenate((zeros,beta_cum_up),axis=1)
			beta_cum_up = np.sum(beta_cum_up_[:,backward_indices[layer_index,1],:]*W_up[1:],axis=2)
			weights_start,weights_stop = weights_stops[layer_index,0],weights_start
			indices = forward_indices[layer_index,1]
			weights_derivate[:,weights_start:weights_stop] = np.hstack((np.sum(beta_cum_up,axis=1),np.sum(beta_cum_up*np.sum(activations_down[indices],axis=1)/ss_window_size,axis=1)))
			W_up = np.reshape(weights[weights_start:weights_stop],(2,n_planes))
			
			activations_down,activations_current = network_activations[layer_index-1,1] if layer_index>0 else activations_down_other_network,activations_down
			
			activations_derivate = activation_derivate_fun(network_activations[layer_index,0][conv_non_paded_index])
			beta_cum_up = beta_cum_up[:,backward_indices[layer_index,0][conv_non_paded_index],:]*activations_derivate*(W_up[1]/ss_window_size)
			weights_start,weights_stop = weights_stops[layer_index-1,1] if layer_index>0 else 0,weights_start
			indices = forward_indices[layer_index,0]
			weights_derivate[:,weights_start:weights_stop] = np.hstack((np.sum(beta_cum_up,axis=1),np.reshape(np.einsum('ijk,jlk->ilk',beta_cum_up,activations_down[indices,:]),(n_units_output,conv_window_size*n_planes))))
			W_up = np.reshape(weights[weights_start:weights_stop],(conv_window_size+1,n_planes))
			activations_current = activations_down

		return None,None


