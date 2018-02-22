import numpy as np
from math import log,log2
from collections import deque
from bisect import bisect_left as bisect
from .base import isclose,less_equal,check_sequence
from .validation import cross_validation
from .myfun import unique

_LESS_EQUAL_THAN_,_LARGER_THAN_,_EQUAL_,_UNEQUAL_ = 0,1,2,3

def print_rules(conditions,predicts,rule_border='-'*40,predict_border='*'*20):
	for condition,predict in zip(conditions,predicts):
		print(rule_border)
		print('CONDITION:')
		fields,values,opses = condition
		for field,value,ops,index in zip(fields,values,opses,range(len(fields))):
			if index>0:
				print('AND\n',end='')
			print('\t',end='')
			print(field,end=' ')
			print('<=' if ops==_LESS_EQUAL_THAN_ else '>' if ops==_LARGER_THAN_ else '==' if ops==_EQUAL_ else '!=',end=' ')
			print('%.6f'%value) if ops<=_LARGER_THAN_ else print(value)
		print(predict_border)
		print('PREDICT:\n\t',predict)
		print(predict_border)

#简化规则
#比如存在规则v>0,v>1，则简化为v>1
def simply_rule(fields,values,ops,attribute_types):
	sorted_indices = np.argsort(fields)
	fields,values,ops = fields[sorted_indices],values[sorted_indices],ops[sorted_indices]
	
	field_names,field_indices,field_counts = unique(fields,return_index=True,return_counts=True,sorted=True)
	n_fields = len(field_names)
	new_fields,new_values,new_ops = [],[],[]

	index_start = 0
	for index in range(n_fields):
		field = field_names[index]
		index_end = field_indices[index+1] if index<n_fields-1 else len(fields)
		values_,ops_ = values[index_start:index_end],ops[index_start:index_end]
		if index_end-index_start>1:
			if attribute_types[field]:
				less_equal_indices,large_indices = ops_==_LESS_EQUAL_THAN_,ops_==_LARGER_THAN_
				if np.any(large_indices):
					new_fields.append(field)
					new_values.append(values_[large_indices].max())
					new_ops.append(_LARGER_THAN_)
				if np.any(less_equal_indices):
					new_fields.append(field)
					new_values.append(values_[less_equal_indices].min())
					new_ops.append(_LESS_EQUAL_THAN_)
			else:
				equal_indices = ops_==_EQUAL_
				if np.any(equal_indices):
					new_fields.append(field)
					new_values.append(values_[equal_indices][0])
					new_ops.append(_EQUAL_)
				else:
					new_fields.extend((field,)*(index_end-index_start))
					new_values.extend(np.sort(values_))
					new_ops.extend((_UNEQUAL_,)*(index_end-index_start))
		else:
			new_fields.append(field)
			new_values.append(values[index_start])
			new_ops.append(ops[index_start])
		index_start = index_end

	return np.array(new_fields,np.object),np.array(new_values,np.object),np.array(new_ops,np.uint8)

#计算各个类别的数据对象的总权重
def calculate_class_weights(target,weights,n_class):
	sorted_indices = np.argsort(target)
	values,value_indices = unique(target,return_index=True,sorted=True)
	class_weights = np.zeros(n_class,np.float64)
	n_values = len(value_indices)
	
	index_start = 0
	for index in range(n_values):
		index_end = value_indices[index+1] if index<n_values-1 else target.shape[0]
		class_weights[values[index]] = np.sum(weights[sorted_indices[index_start:index_end]]) 
		index_start = index_end

	return class_weights

#计算p(j,t)，其中j表示class，t表示结点。见式(2.2)
def get_class_probs(target,n_class,sample_indices,sample_weights,min_sample_weights,totol_class_counts,totol_class_probs,_class_indices_):
	class_counts = np.bincount(target[sample_indices],minlength=n_class).astype(np.float64) if sample_weights is None else calculate_class_weights(target[sample_indices],sample_weights[sample_indices],n_class)
	sum_class_counts = np.sum(class_counts)

	if not sample_weights is None and not min_sample_weights is None and sum_class_counts<min_sample_weights:
		return None,None,None
	
	nz_class_counts_indices = _class_indices_[class_counts>0]
	class_probs = totol_class_probs[nz_class_counts_indices]*class_counts[nz_class_counts_indices]/totol_class_counts[nz_class_counts_indices]
	
	return nz_class_counts_indices,sum_class_counts,class_probs

#train_method_args = (alphas,tree_args)
def cv_train_method(data,target,sample_weights,alphas,tree_args):
	tree = decision_tree(*tree_args)
	tree.train(data,target,sample_weights)
	trees,_ = tree.minimal_cost_complexity_prune(alphas)
	return trees

#predict_method_args = ()
def cv_predict_method(data,trees):
	#print(np.array([tree._root._size for tree in trees],np.uint32))
	return np.array([tree.predict(data) for tree in trees]),np.array([tree._root._size for tree in trees],np.uint32)

#loss_function_args = (n_class,cost_matrix,misclassification_costs)
#计算misclassification_cost[i] = ∑C(i|j)Nij，其中Nij表示属于类j但被分类为类i的数据对象的数量(当sample_weights非None时，Nij表示这些数据对象的总权重)
def cv_loss_function(target,sample_weights,predicts,trees_n_nodes,n_class,cost_matrix,misclassification_costs):
	last_misclassification_cost = np.empty(n_class,np.float64)
	for alpha_index in range(len(predicts)):
		#若相邻两个alpha对应的decision tree相同，跳过
		if alpha_index>0 and trees_n_nodes[alpha_index]==trees_n_nodes[alpha_index-1]:
			misclassification_costs[alpha_index] += last_misclassification_cost
			continue

		predict = predicts[alpha_index]
		misclassification_cost = misclassification_costs[alpha_index]

		misclassified_indices = target!=predict
		misclassified_target = target[misclassified_indices]
		misclassified_predict = predict[misclassified_indices]
		if cost_matrix is None:
			if sample_weights is None:
				last_misclassification_cost = np.bincount(misclassified_target,minlength=n_class)
				misclassification_cost += last_misclassification_cost
			else:
				misclassified_sample_weights = sample_weights[misclassified_indices]
				for class_index in range(n_class):
					last_misclassification_cost[class_index] = np.sum(misclassified_sample_weights[misclassified_target==class_index])
					misclassification_cost[class_index] += last_misclassification_cost[class_index]

		else:
			if sample_weights is None:
				for class_index in range(n_class):
					last_misclassification_cost[class_index] = np.dot(np.bincount(misclassified_predict[misclassified_target==class_index],minlength=n_class),cost_matrix[class_index])
					misclassification_cost[class_index] += last_misclassification_cost[class_index]
			
			else:
				misclassified_sample_weights = sample_weights[misclassified_indices]
				for class_index in range(n_class):
					be_misclassified_indices = misclassified_target==class_index
					if np.sum(be_misclassified_indices) == 0:
						continue
					be_misclassified_target = misclassified_target[be_misclassified_indices]
					misclassified_to_predict = misclassified_predict[be_misclassified_indices]
					be_misclassified_sample_weights = misclassified_sample_weights[be_misclassified_indices]
					for class_other_index in range(n_class):
						if class_index == class_other_index:
							continue
						misclassified_to_indices = misclassified_to_predict==class_other_index
						last_misclassification_cost[class_index] = cost_matrix[class_index,class_other_index]*np.sum(be_misclassified_sample_weights[misclassified_to_indices])
						misclassification_cost[class_index] += last_misclassification_cost[class_index]

	return 0	#无意义

entropy = lambda class_probs:-np.sum(class_probs*np.log(class_probs))
gini_index = lambda class_probs:1-np.sum(class_probs**2)

class decision_tree_node:
	def __init__(self,parent_node,sample_indices,impurity,node_prob,major_class,misclassification_cost):
		self._parent_node,self._child_nodes,self._sample_indices = parent_node,[],sample_indices
		self._attribute_field,self._attribute_values = None,None
		self._impurity,self._impurity_decrease = impurity,None
		self._node_prob,self._major_class,self._misclassification_cost = node_prob,major_class,misclassification_cost
		self._unaccessed,self._R_t,self._R_T,self._g = True,None,None,None
		self._test_node_prob,self._test_misclassification_cost = None,None

	def _split_node(self,data,target,sample_weights,attribute_fields,attribute_types,n_class,leaf_min_n_samples,leaf_min_weights,min_impurity_decrease,cost_matrix,left_most,group_values,is_attribute_names,class_counts,class_probs,_class_indices_,impurity_calculator):
		sample_indices = self._sample_indices
		node_impurity,node_prob = self._impurity,self._node_prob
		max_impurity_decrease = -np.inf
		
		for attribute_field in attribute_fields:
			impurity_decrease,child_node_impurity,attribute_values,child_sample_indices,child_node_probs,child_node_major_class,child_node_misclassification_cost = self._calculate_criteria(data,target,sample_weights,sample_indices,attribute_field,attribute_types[attribute_field],node_impurity,node_prob,n_class,leaf_min_n_samples,leaf_min_weights,min_impurity_decrease,cost_matrix,left_most,group_values,is_attribute_names,class_counts,class_probs,_class_indices_,impurity_calculator)
			if impurity_decrease > max_impurity_decrease:
				max_impurity_decrease,max_child_node_impurity,max_split_attribute_field,max_split_attribute_values,max_child_sample_indices,max_child_node_probs,max_child_node_major_class,max_child_node_misclassification_cost = impurity_decrease,child_node_impurity,attribute_field,attribute_values,child_sample_indices,child_node_probs,child_node_major_class,child_node_misclassification_cost

		if max_impurity_decrease==-np.inf:
			return
		
		self._attribute_field,self._attribute_values = max_split_attribute_field,max_split_attribute_values
		self._impurity_decrease = max_impurity_decrease
		n_child_nodes = len(max_child_sample_indices)
		for child_index in range(n_child_nodes):
			child_node = decision_tree_node(self,max_child_sample_indices[child_index],max_child_node_impurity[child_index],max_child_node_probs[child_index],max_child_node_major_class[child_index],max_child_node_misclassification_cost[child_index])
			self._child_nodes.append(child_node)

	def _calculate_criteria(self,data,target,sample_weights,sample_indices,attribute_field,attribute_type,node_impurity,node_prob,n_class,leaf_min_n_samples,leaf_min_weights,min_impurity_decrease,cost_matrix,left_most,group_values,is_attribute_names,class_counts,class_probs,_class_indices_,impurity_calculator):
		data = data[attribute_field] if is_attribute_names else data[:,attribute_field]
		sample_indices = sample_indices[np.argsort(data[sample_indices])]
		data = data[sample_indices]
		n_samples = data.shape[0]
		sum_class_counts = n_samples if sample_weights is None else np.sum(sample_weights[sample_indices])
		max_impurity_decrease = -np.inf

		#若attribute_type为False，则表示data[attribute_field]存储的是离散数据
		if not attribute_type:
			values,value_indices,value_counts = unique(data,return_index=True,return_counts=True,sorted=True)
			n_values = len(values)
			if n_values<=1 or (group_values and value_counts.max()<leaf_min_n_samples) or (value_counts.min()<leaf_min_n_samples):
				return -np.inf,None,None,None,None,None,None

		if attribute_type or group_values:
			split_indices = range(leaf_min_n_samples,sample_indices.shape[0]-leaf_min_n_samples+1) if attribute_type else range(n_values)
			if not attribute_type:
				split_value_indices,others_value_indices = np.zeros(n_samples,np.bool8),np.ones(n_samples,np.bool8)
				index_start = value_indices[0]

			for split_index in split_indices:
				if attribute_type:
					#存在相同值，跳过
					if isclose(data[split_index-1],data[split_index]):
						continue
					child_left_sample_indices,child_right_sample_indices = sample_indices[:split_index],sample_indices[split_index:]
				else:
					index_end = value_indices[split_index+1] if split_index<n_values-1 else n_samples
					min_child_n_samples = min(index_end-index_start,n_samples-(index_end-index_start))
					if min_child_n_samples<leaf_min_n_samples:
						continue

					split_value_indices[index_start:index_end],others_value_indices[index_start:index_end] = True,False
					child_left_sample_indices,child_right_sample_indices = sample_indices[split_value_indices],sample_indices[others_value_indices]
					split_value_indices[index_start:index_end],others_value_indices[index_start:index_end] = False,True
					index_start = index_end

				nz_class_counts_indices_left,sum_class_counts_left,child_left_class_probs = get_class_probs(target,n_class,child_left_sample_indices,sample_weights,leaf_min_weights,class_counts,class_probs,_class_indices_)
				if nz_class_counts_indices_left is None:
					continue

				nz_class_counts_indices_right,sum_class_counts_right,child_right_class_probs = get_class_probs(target,n_class,child_right_sample_indices,sample_weights,leaf_min_weights,class_counts,class_probs,_class_indices_)
				if nz_class_counts_indices_right is None:
					continue
				
				#计算p(t)，p(t)=∑p(j,t)。见式(2.3)
				child_left_node_probs,child_right_node_probs = np.sum(child_left_class_probs),np.sum(child_right_class_probs)
				
				#计算p(j|t)，p(j|t)=p(j,t)/p(t)。见式(2.4)
				child_left_class_probs /= child_left_node_probs
				child_right_class_probs /= child_right_node_probs
				
				#计算node impurity I(t)
				child_left_impurity,child_right_impurity = impurity_calculator(child_left_class_probs),impurity_calculator(child_right_class_probs)
				
				#impurity decrease = [I(t) - p(tL)I(tL) - p(tR)I(tR)]*p(t)，其中tL,tR分别为结点t的左右子结点。见式(2.7)
				impurity_decrease = ( node_impurity - ( sum_class_counts_left*child_left_impurity + sum_class_counts_right*child_right_impurity )/sum_class_counts )*node_prob
				
				if impurity_decrease > max_impurity_decrease:
					if attribute_type:
						split_attribute_values = data[split_index-1] if left_most else (data[split_index-1]+data[split_index])/2	
					else:
						split_attribute_values = values[split_index]

					max_child_node_impurity = child_left_impurity,child_right_impurity
					max_impurity_decrease = impurity_decrease
					max_child_sample_indices = child_left_sample_indices,child_right_sample_indices
					max_child_node_probs = child_left_node_probs,child_right_node_probs
					max_nz_class_counts_indices_left,max_nz_class_counts_indices_right = nz_class_counts_indices_left,nz_class_counts_indices_right
					max_child_left_class_probs,max_child_right_class_probs = child_left_class_probs,child_right_class_probs

			if max_impurity_decrease != -np.inf:
				if cost_matrix is None:
					max_index_left,max_index_right = np.argmax(max_child_left_class_probs),np.argmax(max_child_right_class_probs)
					max_child_node_major_class = max_nz_class_counts_indices_left[max_index_left],max_nz_class_counts_indices_right[max_index_right]
					#计算misclassification概率，见DEFINITION 2.11
					max_child_node_misclassification_cost = 1-max_child_left_class_probs[max_index_left],1-max_child_right_class_probs[max_index_right]
				else:
					#计算misclassification cost r(t) = min_i ∑C(i|j)p(j|t)，见DEFINITION 2.13
					cost_left,cost_right = np.sum(cost_matrix[max_nz_class_counts_indices_left]*max_child_left_class_probs[:,np.newaxis],axis=0),np.sum(cost_matrix[max_nz_class_counts_indices_right]*max_child_right_class_probs[:,np.newaxis],axis=0)
					min_index_left,min_index_right = np.argmin(cost_left),np.argmin(cost_right)
					max_child_node_major_class = min_index_left,min_index_right
					max_child_node_misclassification_cost = cost_left[min_index_left],cost_right[min_index_right]

		else:
			max_child_node_impurity = np.empty(n_values,np.float64)
			max_child_sample_indices = np.empty(n_values,np.object)
			max_child_node_probs = np.empty(n_values,np.float64)
			max_child_node_major_class = np.empty(n_values,np.uint32)
			max_child_node_misclassification_cost = np.empty(n_values,np.float64)
			
			index_start = value_indices[0]
			sum_child_node_impurity = 0
			for index in range(n_values):
				index_end = value_indices[index+1] if index<n_values-1 else n_samples
				max_child_sample_indices[index] = sample_indices[index_start:index_end]
				index_start = index_end
				
				nz_class_counts_indices,child_sum_class_counts,child_class_probs = get_class_probs(target,n_class,max_child_sample_indices[index],sample_weights,leaf_min_weights,class_counts,class_probs,_class_indices_)
				if nz_class_counts_indices is None:
					break
				
				max_child_node_probs[index] = np.sum(child_class_probs)
				child_class_probs /= max_child_node_probs[index]
				max_child_node_impurity[index] = impurity_calculator(child_class_probs)
				sum_child_node_impurity += child_sum_class_counts*impurity_calculator(child_class_probs)
				
				if cost_matrix is None:
					max_index = np.argmax(child_class_probs)
					max_child_node_major_class[index] = nz_class_counts_indices[max_index]
					max_child_node_misclassification_cost[index] = 1-child_class_probs[max_index]
				else:
					cost = np.sum(cost_matrix[nz_class_counts_indices]*child_class_probs[:,np.newaxis],axis=0)
					min_index = np.argmin(cost)
					max_child_node_major_class[index] = min_index
					max_child_node_misclassification_cost[index] = cost[min_index]
			else:
				max_impurity_decrease = ( node_impurity-sum_child_node_impurity/sum_class_counts )*node_prob
				split_attribute_values = values
		
		if max_impurity_decrease == -np.inf or max_impurity_decrease<min_impurity_decrease:
			return -np.inf,None,None,None,None,None,None
		return max_impurity_decrease,max_child_node_impurity,split_attribute_values,max_child_sample_indices,max_child_node_probs,max_child_node_major_class,max_child_node_misclassification_cost

	#使结点成为叶结点
	def _to_leaf(self):
		access_list = self._child_nodes
		self._child_nodes,self._attribute_field,self._attribute_values,self._impurity_decrease = [],None,None,None
		while len(access_list)>0:
			node = access_list.pop()
			access_list.extend(self._child_nodes)
			del node._child_nodes,node._sample_indices,node._attribute_values
			del node
		
		#更新|Tt|,R_t,R_T,alpha
		update_R_t = not self._R_t is None
		size_diff = 1 - self._size
		self._size = 1
		
		if update_R_t:
			R_T_diff = self._R_t - self._R_T
			self._R_T,self._g = self._R_t,None

		node = self._parent_node
		while not node is None:
			node._size += size_diff
			if update_R_t:
				node._R_T += R_T_diff
				node._g = (node._R_t - node._R_T) / (node._size - 1)

			node = node._parent_node
		
class decision_tree:
	def __init__(self,n_class,attribute_types,impurity_calculator='entropy',cost_matrix=None,min_impurity_decrease=None,leaf_min_n_samples=None,leaf_min_weights=None,left_most=False,group_values=True):
		"""
			Decision Tree。目前只支持分类决策树。具体见论文<<Classification and Regression Trees>>
			参数：
				①n_class：整型。表示类的数量
				②attribute_types：sequence或dict，表示各个属性是连续型还是离散型。当attribute_types为dict时，其键表示属性名，其值为bool，当对应属性为连续型，则值为True，否则为False
				③impurity_calculator：str或callable object，表示用于计算结点impurity的函数。当impurity_calculator为'entropy'时，则结点的impurity为其entropy(此时impurity decrease则为imformation gain)。当impurity_calculator为'gini'时，则结点的impurity为其gini index
				④cost_matrix：shape为(n_class,n_class)的2D array。cost_matrix[i,j]表示C(i|j)，即类别为j时被误判为类别i的损失。cost_matrix默认为None，即使用unit cost，此时cost_matrix的对角元素为0，其余元素为1
				⑤min_impurity_decrease：浮点数，表示分割非叶结点时，最小所需的impurity decrease。若分割某一结点造成的impurity decrease小于min_impurity_decrease，则停止分割，该结点成为叶结点。因此min_impurity_decrease可用于pre-pruning。默认为None，即0.0
				⑥leaf_min_n_samples：整型，表示分割非叶结点时，每个子结点最小所包含的数据对象数量。若分割某一结点，其某一子结点包含的数据对象数量小于leaf_min_n_samples，则停止分割，该结点成为叶结点。因此leaf_min_n_samples可用于pre-pruning。默认为None，即1
				⑦leaf_min_weights：浮点数，表示分割非叶结点时，每个子结点包含的数据对象的最小所需总权重。若分割某一结点，其某一子结点包含的数据对象的总权重小于leaf_min_weights，则停止分割，该结点成为叶结点。因此leaf_min_weights可用于pre-pruning。默认为None，即0.0
				⑧group_values：bool。当在结点上分割离散型属性时，若group_values为True，则将结点一分为二(=与!=)，否则一份为多。默认为True
		"""
		if impurity_calculator == 'entropy':
			self._impurity_calculator = entropy
		elif impurity_calculator == 'gini':
			self._impurity_calculator = gini_index
		elif callable(impurity_calculator):
			self._impurity_calculator = impurity_calculator
		else:
			raise ValueError("impurity_calculator must be str like 'entropy' or 'gini',or callable object")

		if min_impurity_decrease is None:
			min_impurity_decrease = 0.0
		if leaf_min_n_samples is None:
			leaf_min_n_samples = 1
		if leaf_min_weights is None:
			leaf_min_weights = 0

		self._cost_matrix = cost_matrix
		self._n_class = n_class
		self._min_impurity_decrease = min_impurity_decrease
		self._leaf_min_n_samples,self._leaf_min_weights = leaf_min_n_samples,leaf_min_weights
		self._attribute_types = attribute_types
		self._left_most,self._group_values = left_most,group_values
		self._attribute_fields,self._is_attribute_names = (attribute_types.keys(),True) if type(attribute_types)==dict else (np.arange(len(attribute_types),dtype=np.uint32),False)
	
	def train(self,data,target,sample_weights=None,class_probs=None):
		"""
			训练决策树。
			参数：
				①data,target：训练集。target取值范围必须为{0,1,...,n_class-1}
				②sample_weights：1D array，表示各个数据对象的权重。若sample_weights为None，则认为所有数据对象具有相同权重(都为1)
				③class_probs：1D array，表示各个类别的先验概率。若class_probs为None，则从训练集获取各个类别的出现频率作为先验概率
		"""
		class_counts = np.bincount(target) if sample_weights is None else calculate_class_weights(target,sample_weights,self._n_class)
		if class_probs is None:
			class_probs = class_counts.astype(np.float64)
		class_probs /= np.sum(class_probs)
		
		major_class = np.argmax(class_counts)
		self._root = decision_tree_node(None,np.arange(len(data),dtype=np.uint32),self._impurity_calculator(class_probs),1.0,major_class,1-class_counts[major_class]/len(data))
		_class_indices_ = np.arange(self._n_class,dtype=np.uint32)

		access_list = [self._root]
		while len(access_list)>0:
			node = access_list.pop()
			node._split_node(data,target,sample_weights,self._attribute_fields,self._attribute_types,self._n_class,self._leaf_min_n_samples,self._leaf_min_weights,self._min_impurity_decrease,self._cost_matrix,self._left_most,self._group_values,self._is_attribute_names,class_counts,class_probs,_class_indices_,self._impurity_calculator)
			access_list.extend(node._child_nodes)
		
		#计算各个结点的size，即结点包含的后代叶结点的数量|Tt|(DEFINITION 3.5)
		access_list = [self._root]
		while len(access_list)>0:
			node = access_list[-1]
			child_nodes = node._child_nodes
			#叶节点
			if len(child_nodes)==0:
				access_list.pop()
				node._size = 1

			#被叶结点第一次被访问
			elif node._unaccessed:
				access_list.extend(child_nodes)
				node._unaccessed = False

			else:
				access_list.pop()
				node._unaccessed = True
				node._size = sum([child_node._size for child_node in child_nodes])

		self._data,self._target,self._sample_weights = data,target,sample_weights
		self._class_counts,self._class_probs = class_counts,class_probs

	def predict(self,data):
		"""
			预测数据集data的分类
		"""
		terminal_major_class,terminal_sample_indices = self._travel(data)
		predict = np.empty(len(data),np.uint16)
		for index in range(len(terminal_major_class)):
			predict[terminal_sample_indices[index]] = terminal_major_class[index]
		return predict

	def to_rules(self,variable_names=None):
		"""
			show rules:
				>>>print_rules(*tree.to_rules())
		"""
		root = self._root
		if root._size == 1:
			return None,None
		attribute_types,is_attribute_names,group_values = self._attribute_types,self._is_attribute_names,self._group_values
		
		if not is_attribute_names:
			if check_sequence(variable_names):
				variable_names = [str(variable_name) for variable_name in variable_names]
			elif type(variable_names)==str:
				variable_names = ['%s_'%variable_names+'%d'%i for i in range(len(attribute_types))]
			else:
				variable_names = ['variable_'+'%d'%i for i in range(len(attribute_types))]
			variable_names = np.array(variable_names)
			_attribute_types_ = dict([(variable_names[index],attribute_types[index]) for index in range(len(attribute_types))])
		else:
			_attribute_types_ = attribute_types
		
		index = 0
		conditions,predicts = np.empty(root._size,np.object),np.empty(root._size,np.uint16)
		access_list = [(root,[],[],[])]
		while len(access_list)>0:
			node,fields,values,ops = access_list.pop()
			child_nodes = node._child_nodes
			if len(child_nodes)==0:
				fields,values,ops = np.array(fields,np.object) if is_attribute_names else variable_names[fields],np.array(values,np.object),np.array(ops,np.uint8)
				conditions[index] = simply_rule(fields,values,ops,_attribute_types_)
				predicts[index] = node._major_class
				index += 1
				continue

			attribute_field,attribute_values = node._attribute_field,node._attribute_values
			fields.append(attribute_field)
			if attribute_types[attribute_field] or group_values:
				values.append(attribute_values)
				child_left_fields,child_right_fields = fields,fields.copy()
				child_left_values,child_right_values = values,values.copy()
				
				child_left_ops,child_right_ops = ops,ops.copy()
				#ops为_LESS_EQUAL_THAN_代表"<="，ops为_EQUAL_代表"=="
				child_left_ops.append(_LESS_EQUAL_THAN_ if attribute_types[attribute_field] else _EQUAL_)
				#ops为1代表"_LARGER_THAN_"，ops为3代表"_UNEQUAL_"
				child_right_ops.append(_LARGER_THAN_ if attribute_types[attribute_field] else _UNEQUAL_)
				
				access_list.extend(list(zip(child_nodes,(child_left_fields,child_right_fields),(child_left_values,child_right_values),(child_left_ops,child_right_ops)))[::-1])
			
			else:
				ops.append(_EQUAL_)
				for child_index in range(len(child_nodes)-1,-1,-1):
					fields_,values_,ops_ = fields.copy(),values.copy(),ops.copy()
					values_.append(attribute_values[child_index])
					access_list.append((child_nodes[child_index],fields_,values_,ops_))

		return conditions,predicts

	#目前并不支持group_values=False的情形(然而支持group_values并不难达到，待完成......)
	def variable_importance(self):
		"""
			计算各个属性的importance，见5.3.4
		"""
		if not self._group_values:
			raise ValueError('multiway tree does not support variable importance,change group_values to True')
		if not hasattr(self,'_data'):
			if self._root is None:
				raise ValueError('train first')
			else:
				raise ValueError('structure-only tree does not support variable importance')

		n_class,impurity_calculator = self._n_class,self._impurity_calculator
		attribute_types,is_attribute_names = self._attribute_types,self._is_attribute_names
		data,target,sample_weights = self._data,self._target,self._sample_weights
		class_counts,class_probs = self._class_counts,self._class_probs

		attribute_fields = attribute_types.keys() if self._is_attribute_names else range(len(attribute_types))
		variable_importances = dict((key,0) for key in attribute_fields) if is_attribute_names else np.zeros(len(attribute_fields),np.float64)
		_class_indices_ = np.arange(n_class,dtype=np.uint32)

		access_list = [self._root]
		while len(access_list)>0:
			node = access_list.pop()
			if len(node._child_nodes)==0:
				continue
			access_list.extend(node._child_nodes)
			
			child_left_node,child_right_node = node._child_nodes
			sample_indices = node._sample_indices
			node_impurity,node_prob = node._impurity,node._node_prob
			n_samples = sample_indices.shape[0]
			sum_class_counts = n_samples if sample_weights is None else np.sum(sample_weights[sample_indices])
		
			for attribute_field in attribute_fields:
				data_ = data[attribute_field] if is_attribute_names else data[:,attribute_field]
				sample_indices = sample_indices[np.argsort(data_[sample_indices])]
				data_ = data_[sample_indices]
				attribute_type = attribute_types[attribute_field]

				if attribute_type:
					split_indices = range(1,n_samples)
				else:
					values,value_indices = unique(data_,return_index=True,sorted=True)
					n_values = len(values)
					if n_values<=1:
						continue

					split_value_indices,others_value_indices = np.zeros(n_samples,np.bool8),np.ones(n_samples,np.bool8)
					index_start = value_indices[0]
					split_indices = range(n_values)

				max_surrogate_prob,max_impurity_decrease = -np.inf,-np.inf
				for split_index in split_indices:
					if attribute_type:
						if isclose(data_[split_index-1],data_[split_index]):
							continue
						child_left_sample_indices,child_right_sample_indices = sample_indices[:split_index],sample_indices[split_index:]
					else:
						index_end = value_indices[split_index+1] if split_index<n_values-1 else n_samples
						split_value_indices[index_start:index_end],others_value_indices[index_start:index_end] = True,False
						child_left_sample_indices,child_right_sample_indices = sample_indices[split_value_indices],sample_indices[others_value_indices]
						split_value_indices[index_start:index_end],others_value_indices[index_start:index_end] = False,True
						index_start = index_end
					
					common_indices_left,common_indices_right = np.intersect1d(child_left_sample_indices,child_left_node._sample_indices),np.intersect1d(child_right_sample_indices,child_right_node._sample_indices)
					
					#surrogate_prob的计算见5.3.1
					_,_,child_left_class_probs = get_class_probs(target,n_class,common_indices_left,sample_weights,None,class_counts,class_probs,_class_indices_)
					_,_,child_right_class_probs = get_class_probs(target,n_class,common_indices_right,sample_weights,None,class_counts,class_probs,_class_indices_)
					surrogate_prob = (np.sum(child_left_class_probs) + np.sum(child_right_class_probs)) / node_prob

					if surrogate_prob < max_surrogate_prob:
						continue

					#计算该surrogate对应的impurity decrease，同decision_tree_node._calculate_criteria
					_,sum_class_counts_left,child_left_class_probs = get_class_probs(target,n_class,child_left_sample_indices,sample_weights,None,class_counts,class_probs,_class_indices_)
					_,sum_class_counts_right,child_right_class_probs = get_class_probs(target,n_class,child_right_sample_indices,sample_weights,None,class_counts,class_probs,_class_indices_)
					child_left_class_probs /= np.sum(child_left_class_probs)
					child_right_class_probs /= np.sum(child_right_class_probs)
					
					child_left_impurity,child_right_impurity = impurity_calculator(child_left_class_probs),impurity_calculator(child_right_class_probs)
					impurity_decrease = ( node_impurity - ( sum_class_counts_left*child_left_impurity + sum_class_counts_right*child_right_impurity )/sum_class_counts )*node_prob
					
					if max_surrogate_prob<surrogate_prob or impurity_decrease > max_impurity_decrease:
						max_surrogate_prob = surrogate_prob
						max_impurity_decrease = impurity_decrease
			
				if max_impurity_decrease != -np.inf:
					variable_importances[attribute_field] += max_impurity_decrease
		return variable_importances
	
	#目前simple_prune是在训练集或测试集上进行剪枝
	#但完全可以拓展到验证集的剪枝
	#待完成
	def simple_prune(self,data=None,target=None):
		"""
			error_based pruning。若不给定测试集(data=None,target=None)，则在训练集上进行剪枝，否则在测试集上进行剪枝
		"""
		if not data is None and not target is None:
			self._travel(data,target)
			test = True
		else:
			test = False

		self._calculate_R(test)

		access_list = [self._root]
		while len(access_list)>0:
			node = access_list.pop()

			if isclose(node._R_t,node._R_T):
				node._to_leaf()
			elif len(node._child_nodes)>0:
				access_list.extend(node._child_nodes)
			
	#若给定alphas，则获取当complexity parameter α取alphas中各个值时的decision tree
	#目前minimal cost-complexity pruning只实现了在训练集和验证集的剪枝
	#但完全可以拓展到在测试集上的剪枝，见论文3.4.1节
	#待完成
	def minimal_cost_complexity_prune(self,alphas=None):
		"""
			minimal cost-complexity pruning，见论文3.3节
		"""
		trees = [self._copy_structure()]
		pre_tree = trees[-1]
		
		pre_tree.simple_prune()
		alpha_jump_points = [0]
		
		max_alpha = np.inf if alphas is None else alphas[-1]
		while pre_tree._root._size>1:
			new_tree = pre_tree._copy_structure()
			
			min_g = np.inf
			access_list = [new_tree._root]
			while len(access_list)>0:
				node = access_list.pop()
				if len(node._child_nodes)>0:
					access_list.extend(node._child_nodes)
					if node._g<min_g:
						min_g,min_node = node._g,node
			
			if min_g>max_alpha:
				break
			min_node._to_leaf()
			pre_tree = new_tree
			
			alpha_jump_points.append(min_g)
			trees.append(new_tree)
		
		#假设我们已经获取了complexity parameter取alpha_jump_points中各个值时的decision tree，现在想获取当complexity parameter α取alphas中各个值时的decision tree
		#令α1、α2为alpha_jump_points中任意两个相邻的值，且α1<α2。由于当α1<=α<α2时，complexity parameter为α的decision tree与complexity parameter为α1的decision tree完全相同
		#为了获取complexity parameter α取alphas中某个值alpha时的decision tree，我们找出alpha_jump_points中小于等于alpha的最大值，其对应的decision即为所求decision tree
		if not alphas is None:
			trees_ = []
			current_alphas_index,current_alpha_jump_points_index = 0,0
			current_alpha,current_alpha_jump_point = alphas[0],alpha_jump_points[0]
			n_alphas,n_alpha_jump_points = len(alphas),len(alpha_jump_points)
			while current_alphas_index<n_alphas and current_alpha_jump_points_index<n_alpha_jump_points:
				if less_equal(current_alpha,current_alpha_jump_point):
					current_alphas_index += 1
					current_alpha = alphas[current_alphas_index] if current_alphas_index<n_alphas else 0
					trees_.append(trees[current_alpha_jump_points_index])
				elif less_equal(current_alpha_jump_point,current_alpha):
					current_alpha_jump_points_index +=1
					current_alpha_jump_point = alpha_jump_points[current_alpha_jump_points_index] if current_alpha_jump_points_index<n_alpha_jump_points else 0

			if current_alphas_index!=n_alphas:
				trees_.extend([trees[-1] for _ in range(n_alphas-current_alphas_index)])
			trees = trees_

		return np.array(trees,np.object),np.array(alpha_jump_points if alphas is None else alphas,dtype=np.float64)
	
	#利用交叉验证来进行剪枝，主要是利用交叉验证来获取Rcv(T(α))，即利用交叉验证来获取complexity parameter为α的decision tree的misclassification cost
	def cv_minimal_cost_complexity_prune(self,n_fold=5):
		"""
			minimal cost-complexity pruning using cross-validation estimates。见论文3.4.2节
		"""
		trees,alphas = self.minimal_cost_complexity_prune()
		
		tree_args = (self._n_class,self._attribute_types,self._impurity_calculator,self._cost_matrix,self._min_impurity_decrease,self._leaf_min_n_samples,self._leaf_min_weights,self._left_most,self._group_values)
		
		cv_alphas = alphas.copy()
		#取alphas相邻两个值的几何平均值，见3.4.2
		cv_alphas[:-1] = np.sqrt(cv_alphas[:-1]*cv_alphas[1:])
		misclassification_costs = np.zeros((alphas.shape[0],self._n_class),np.float64)
		
		train_method_args = (cv_alphas,tree_args)
		predict_method_args = ()
		loss_function_args = (self._n_class,self._cost_matrix,misclassification_costs)

		cross_validation(self._data,self._target,self._sample_weights,n_fold,cv_train_method,cv_predict_method,cv_loss_function,None,train_method_args,predict_method_args,loss_function_args)

		#Rcv(T(α))=∑Rcv(j)π(j)，其中j代表类别，π(j)表示类别j的先验概率。计算见式(3.16)
		Rcv = np.sum(misclassification_costs/self._class_counts*self._class_probs,axis=1)
		return trees,alphas,Rcv

	def _copy_structure(self):
		new_tree = decision_tree(self._n_class,self._attribute_types,self._impurity_calculator,self._cost_matrix,self._min_impurity_decrease,self._leaf_min_n_samples,self._leaf_min_weights,self._left_most,self._group_values)
		
		root = self._root
		new_tree_root = decision_tree_node(None,None,root._impurity,root._node_prob,root._major_class,root._misclassification_cost)
		new_tree_root._attribute_field,new_tree_root._attribute_values,new_tree_root._impurity_decrease = root._attribute_field,root._attribute_values,root._impurity_decrease
		new_tree_root._size,new_tree_root._R_t,new_tree_root._R_T,new_tree_root._g = root._size,root._R_t,root._R_T,root._g
		new_tree._root = new_tree_root

		access_list = [(root,new_tree_root)]
		while len(access_list)>0:
			node,new_tree_node = access_list.pop()
			child_nodes,new_tree_child_nodes = node._child_nodes,new_tree_node._child_nodes

			if len(child_nodes)>0:
				for child_node in child_nodes:
					new_tree_child_node = decision_tree_node(new_tree_node,None,child_node._impurity,child_node._node_prob,child_node._major_class,child_node._misclassification_cost)
					new_tree_child_node._attribute_field,new_tree_child_node._attribute_values,new_tree_child_node._impurity_decrease = child_node._attribute_field,child_node._attribute_values,child_node._impurity_decrease
					new_tree_child_node._size,new_tree_child_node._R_t,new_tree_child_node._R_T,new_tree_child_node._g = child_node._size,child_node._R_t,child_node._R_T,child_node._g
					new_tree_child_nodes.append(new_tree_child_node)
				
				access_list.extend(zip(child_nodes,new_tree_child_nodes))
		
		return new_tree

	def _travel(self,data,target=None):
		if target is None:
			terminal_major_class,terminal_sample_indices = [],[]
		else:
			n_class,cost_matrix = self._n_class,self._cost_matrix
		
		n_samples = data.shape[0]
		attribute_types,is_attribute_names,group_values = self._attribute_types,self._is_attribute_names,self._group_values
		access_list = [(self._root,np.arange(len(data),dtype=np.uint32))]
		while len(access_list)>0:
			node,sample_indices = access_list.pop()
			if sample_indices.shape[0] == 0:
				continue

			if not target is None:
				misclassified_indices = target[sample_indices] != node._major_class
				node._test_node_prob = sample_indices.shape[0]/n_samples
				if cost_matrix is None:
					node._test_misclassification_cost = np.sum(misclassified_indices)/sample_indices.shape[0]
				else:
					misclassified_class_probs = np.bincount(target[misclassified_indices],minlength=n_class)/sample_indices.shape[0]
					node._test_misclassification_cost = np.dot(cost_matrix[node._major_class],misclassified_class_probs)

			if len(node._child_nodes)==0:
				if target is None:
					terminal_major_class.append(node._major_class)
					terminal_sample_indices.append(sample_indices)
				continue

			attribute_type,attribute_field,attribute_values = attribute_types[node._attribute_field],node._attribute_field,node._attribute_values
			data_ = data[attribute_field][sample_indices] if is_attribute_names else data[sample_indices,attribute_field]
			
			if attribute_type or group_values:
				bool_indices = np.less_equal(data_,attribute_values) if attribute_type else data_==attribute_values
				child_sample_indices = sample_indices[bool_indices],sample_indices[np.logical_not(bool_indices)]
				access_list.extend(zip(node._child_nodes,child_sample_indices))
				
			else:
				child_nodes = node._child_nodes
				for index in range(len(attribute_values)):
					access_list.append((child_nodes[index],sample_indices[data_==attribute_values[index]]))

		if target is None:
			return terminal_major_class,terminal_sample_indices
	
	#计算misclassification cost of tree T R(T)与misclassification cost of node t R(t)
	#见DEFINITION 2.13
	#g的定义见式(3.9)
	def _calculate_R(self,test=False):
		access_list = [self._root]
		while len(access_list)>0:
			node = access_list[-1]
			child_nodes = node._child_nodes

			#叶节点
			if len(child_nodes)==0:
				access_list.pop()
				if test and node._test_node_prob is None:
					continue
				node._R_T = node._R_t = node._test_node_prob*node._test_misclassification_cost if test else node._node_prob*node._misclassification_cost
				node._test_node_prob,node._test_misclassification_cost = None,None

			#非叶结点第一次被访问
			elif node._unaccessed:
				access_list.extend(child_nodes)
				node._unaccessed = False

			else:
				access_list.pop()
				node._unaccessed = True
				if test and node._test_node_prob is None:
					continue

				node._R_t = node._test_node_prob*node._test_misclassification_cost if test else node._node_prob*node._misclassification_cost
				node._test_node_prob,node._test_misclassification_cost = None,None
				
				sum_R_T,has_None = 0.0,False
				for child_node in child_nodes:
					if child_node._R_T is None:
						has_None = True
						break
					sum_R_T += child_node._R_T

				if has_None:
					continue
				
				node._R_T = sum_R_T
				if not test:
					node._g = (node._R_t - node._R_T) / (node._size - 1)

