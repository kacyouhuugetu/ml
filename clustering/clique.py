from math import log2
from itertools import combinations
from bisect import bisect_left,bisect_right
from ..base import sort_union,sort_difference,sort_intersection,sort_search,sort_search_continuous_left_most,sort_search_continuous_right_most,partition_data_space
import numpy as np

def remove_redundant_regions(regions):
	n_regions = len(regions)
	sorted_indices = np.argsort([len(region) for region in regions])
	sorted_regions,region_sets = [],[]
	
	for region_index in sorted_indices:
		sorted_regions.append(regions[region_index])
		region_sets.append(set([tuple(ele) for ele in regions[region_index]]))

	overlay_region = set()
	pop_indices = []
	for region_index in range(n_regions-1):
		region = region_sets[region_index]
		overlay_region = set()
		#removal heuristic
		#获取当前region与其它所有regions的交集(overlay_region)，若该交集等于region，则表示region被其它的regions所覆盖，因此删除该region
		for another_region_index in range(region_index+1,n_regions):
			overlay_region.update(region.intersection(region_sets[another_region_index]))
		
		if overlay_region==region:
			pop_indices.append(sorted_indices[region_index])

	return pop_indices

#计算各个子空间的code length，返回要移除的子空间的索引
#参照论文<<Automatic Subspace Clustering of High Dimensional Data for Data Mining Applications>>的3.1.2节
def mdl_interest(subspaces_itemsets_count,least_keep_ratio=1.0):
	n_subspaces = len(subspaces_itemsets_count)
	least_keep_ratio = max(0,min(1.0,least_keep_ratio))
	n_keep_subspaces = int(max(2,n_subspaces*least_keep_ratio))
	
	subspaces_coverage = np.array([sum([sum(cluster_itemsets_count) for cluster_itemsets_count in subspace_itemsets_count]) for subspace_itemsets_count in subspaces_itemsets_count],np.uint32)
	sorted_indices = np.argsort(-subspaces_coverage)
	
	min_cl,min_index = np.inf,n_subspaces
	for index in range(n_keep_subspaces,n_subspaces-1):
		keep_subspaces_coverage,prune_subspaces_coverage = subspaces_coverage[sorted_indices[:index]],subspaces_coverage[sorted_indices[index:]]
		mu_i,mu_p = np.sum(keep_subspaces_coverage)/index,np.sum(prune_subspaces_coverage)/(n_subspaces-index)
		if np.any(np.isclose(keep_subspaces_coverage,mu_i,atol=1e-15)) or np.any(np.isclose(prune_subspaces_coverage,mu_p,atol=1e-15)):
			continue

		cl = np.sum(np.log2(np.abs(keep_subspaces_coverage-mu_i))) + np.sum(np.log2(np.abs(prune_subspaces_coverage-mu_p))) + log2(mu_i) + log2(mu_p)
		if cl<min_cl:
			min_cl,min_index = cl,index
	
	return np.sort(sorted_indices[min_index:]).tolist()

#计算各个子空间的entropy，返回要移除的子空间的索引
#参照论文<<Entropy-based Subspace Clustering for Mining Numerical Data>>的4.1节
def entropy_interest(subspaces_itemsets_count,max_entropy=None,least_keep_ratio=1.0):
	max_entropy = np.inf if max_entropy is None else max_entropy
	n_subspaces = len(subspaces_itemsets_count)

	subspaces_entropy,pop_indices = [],[]
	for subspace_index in range(n_subspaces):
		probs = np.hstack(subspaces_itemsets_count[subspace_index]).astype(np.float64)
		probs /= np.sum(probs)

		subspace_entropy = -np.sum(probs*np.log(probs))
		if subspace_entropy < max_entropy:
			subspaces_entropy.append(subspace_entropy)
		else:
			pop_indices.append(subspace_index)

	n_subspaces = len(subspaces_entropy)
	least_keep_ratio = max(0,min(1.0,least_keep_ratio))
	n_keep_subspaces = int(n_subspaces*least_keep_ratio)

	sorted_indices = np.argsort(subspaces_entropy)
	pop_indices = np.hstack((pop_indices,sorted_indices[n_keep_subspaces:])).astype(np.uint32)
	
	return np.sort(pop_indices).tolist()

class trie_node:
	def __init__(self,level,item,parent_node):
		self._level,self._item,self._parent_node = level,item,parent_node
		self._child_nodes,self._E,self._AE,self._F,self._projected_T = [],[],[],[],[]
		self._count,self._accessed = 0,False
	
	#axes表示该结点的所有extension所属的axis的集合
	#axis_index_of_items表示该结点各个extension所属的axis在该结点的axes中的索引
	#items_in_axes表示各个属于各个axis的extension
	#axes_start_index用于获取某一axis的某一extension在该结点的extensions(变量_E)中的索引，假设某一extension在其所属的axis的索引为i，而该axis对应的axis_start_index为j，则该extension在结点的extensions的索引为i+j
	def _gen_axes(self,knots):
		items = self._AE
		if len(items)==0:
			return
		
		axis_of_items = np.searchsorted(knots,items,side='right')-1
		axes,indices = np.unique(axis_of_items,return_index=True,sorted=True)
		n_axes = axes.shape[0]

		self._axes = axes.tolist()
		self._axis_index_of_items = np.searchsorted(axes,axis_of_items,side='left').tolist()
		self._axes_start_index,self._items_in_axes = [],[]
		
		start_index = 0
		for axis_index in range(n_axes):
			end_index = indices[axis_index+1] if axis_index<n_axes-1 else len(items)
			self._axes_start_index.append(start_index)
			self._items_in_axes.append(items[start_index:end_index])
			start_index = end_index
	
	#沿着从结点点到根结点的路径，生成itemset
	def _get_itemset(self):
		itemset = []
		node = self
		parent_node = node._parent_node
		while not parent_node is None:
			itemset.append(node._item)
			node,parent_node = parent_node,parent_node._parent_node
		itemset.reverse()
		return tuple(itemset)
	
	#get ancestor of level n
	def _get_ancestor(self,n):
		node = self
		while node._level>n:
			node = node._parent_node
		return node

	def _search_path(self,search_items):
		#if self._level+len(search_items)>k(maximal level of tree),we could exit and return None
		node = self
		for item in search_items:
			#when self._level+len(search_items)==k(maximal level of tree),it would be better to use node._AE instead of node._E
			item_index_in_E = sort_search(node._E,item)
			if item_index_in_E is None:
				return None

			node = node._child_nodes[item_index_in_E]
		
		return node

	def _delete_node(self,knots=None,itemset_axes=None):
		if len(self._child_nodes)>0:
			raise ValueError("Error:the node to be deleted must be a leaf node")
		if knots is None and itemset_axes is None:
			raise ValueError("Error:knots and itemset_itervals are both None")

		node = self
		parent_node = node._parent_node
		index = -1
		while not parent_node is None:
			item = node._item
			
			E,AE,F = parent_node._E,parent_node._AE,parent_node._F
			child_nodes = parent_node._child_nodes
			axes,axes_start_index,axis_index_of_items,items_in_axes = parent_node._axes,parent_node._axes_start_index,parent_node._axis_index_of_items,parent_node._items_in_axes
			
			axis = bisect_right(knots,item)-1 if itemset_axes is None else itemset_axes[index]
			item_axis_index = bisect_left(axes,axis)
			item_index_in_axis = bisect_left(items_in_axes[item_axis_index],item)
			item_index_in_E = axes_start_index[item_axis_index] + item_index_in_axis
			
			E.pop(item_index_in_E)
			child_nodes.pop(item_index_in_E)
			axis_index_of_items.pop(item_index_in_E)
			items_in_axes[item_axis_index].pop(item_index_in_axis)
			for axis_index in range(item_axis_index+1,len(axes)):
				axes_start_index[axis_index] -= 1
			
			item_index_in_AE = sort_search(AE,item)
			if not item_index_in_AE is None:
				AE.pop(item_index_in_AE)

			F.clear()
			for child_node in child_nodes:
				item_index_in_AE = sort_search(AE,child_node._item)
				if not item_index_in_AE is None:
					sort_union(F,child_node._F,inplace=True)

			if len(items_in_axes[item_axis_index])==0:
				axes.pop(item_axis_index)
				axes_start_index.pop(item_axis_index)
				items_in_axes.pop(item_axis_index)
				for item_index in range(item_index_in_E,len(E)):
					axis_index_of_items[item_index] -= 1

				if len(axes)==0:
					node,parent_node = parent_node,parent_node._parent_node
					index -= 1
					continue
			parent_node = None

	#获取所有与该结点具有common face的结点
	#由于对数据空间进行了partition，则两个结点具有common face意味着两个结点的itemset的只有一对item的绝对差值为1，而其它item相同
	def _get_neighbor(self,root,return_itemsets=True,visit_unaccessed=True,set_accessed=True,itemset=None):
		if itemset is None:
			itemset = self._get_itemset()
		n_items = len(itemset)

		neighbor_nodes = []
		if return_itemsets:
			neighbor_itemsets = []
		for item_index in range(n_items):
			item,prefix_items,suffix_items = itemset[item_index],itemset[:item_index],itemset[item_index+1:]
			
			prefix_node = self._get_ancestor(item_index)
			item_index_in_E = bisect_left(prefix_node._E,item)
			item_axis_index = prefix_node._axis_index_of_items[item_index_in_E]
			item_axis_start_index = prefix_node._axes_start_index[item_axis_index]
			items = prefix_node._items_in_axes[item_axis_index]
			
			left_neighbor_index,right_neighbor_index = sort_search(items,item-1),sort_search(items,item+1)
			if not left_neighbor_index is None:
				left_child_node = prefix_node._child_nodes[item_axis_start_index+left_neighbor_index]
				suffix_node = left_child_node._search_path(suffix_items)
				if not suffix_node is None and suffix_node._accessed!=visit_unaccessed:
					if set_accessed:
						suffix_node._accessed = visit_unaccessed
					neighbor_nodes.append(suffix_node)
					if return_itemsets:
						neighbor_itemsets.append(prefix_items+(item-1,)+suffix_items)

			if not right_neighbor_index is None:
				right_child_node = prefix_node._child_nodes[item_axis_start_index+right_neighbor_index]
				suffix_node = right_child_node._search_path(suffix_items)
				if not suffix_node is None and suffix_node._accessed!=visit_unaccessed:
					if set_accessed:
						suffix_node._accessed = visit_unaccessed
					neighbor_nodes.append(suffix_node)
					if return_itemsets:
						neighbor_itemsets.append(prefix_items+(item+1,)+suffix_items)
		if return_itemsets:
			return neighbor_nodes,neighbor_itemsets
		else:
			return neighbor_nodes

	#获取从结点开始，所有连接结点(connected)的集合
	def _get_cluster(self,root,itemset=None,return_itemsets=True,return_nodes=False,return_count=False):
		itemset = self._get_itemset() if itemset is None else itemset
		cluster_itemsets,cluster_nodes,cluster_itemsets_count = None,None,None
		if return_itemsets:
			cluster_itemsets = [itemset]
		if return_nodes:
			cluster_nodes = [self]
		if return_count:
			cluster_itemsets_count = [self._count]

		if self._level>1 and not self._accessed:
			access_list = [(self,itemset)]
			self._accessed = True
			while len(access_list)>0:
				node,itemset = access_list.pop()

				neighbor_nodes,neighbor_itemsets = node._get_neighbor(root,itemset=itemset)
				if len(neighbor_nodes)>0:
					access_list.extend(zip(neighbor_nodes,neighbor_itemsets))
					if return_itemsets:
						cluster_itemsets.extend(neighbor_itemsets)
					if return_nodes:
						cluster_nodes.extend(neighbor_nodes)
					if return_count:
						cluster_itemsets_count.extend([node._count for node in neighbor_nodes])

		return cluster_itemsets,cluster_nodes,cluster_itemsets_count

	def _reset_accessed(self,root):
		access_list = [self]
		self._accessed = False
		while len(access_list)>0:
			node = access_list.pop()

			neighbor_nodes = node._get_neighbor(root,return_itemsets=False,visit_unaccessed=False)
			if len(neighbor_nodes)>0:
				access_list.extend(neighbor_nodes)

	def _get_maximal_region(self,root,itemset=None):
		if self._accessed:
			return None,None,None,None,None,None
		if itemset is None:
			itemset = self._get_itemset()
		self._accessed = True

		region_nodes,region_left_width,region_right_width,region_itemsets = [self],[],[],[itemset]
		centre_itemset = itemset
		n_items = len(itemset)
		for item_index in range(n_items):
			left_width,right_width = np.inf,np.inf
			len_region_nodes = len(region_nodes)
			extend_left_nodes,extend_right_nodes,extend_left_itemsets,extend_right_itemsets = [[] for _ in range(len_region_nodes)],[[] for _ in range(len_region_nodes)],[[] for _ in range(len_region_nodes)],[[] for _ in range(len_region_nodes)]
			
			extend_left,extend_right = True,True
			for node_index in range(len_region_nodes):
				node,itemset = region_nodes[node_index],region_itemsets[node_index]
				item,prefix_items,suffix_items = itemset[item_index],itemset[:item_index],itemset[item_index+1:]
				prefix_node = node._get_ancestor(item_index)
				child_nodes = prefix_node._child_nodes

				#when self._level==k(maximal level of tree),it would be better to use prefix_node._AE instead of prefix_node._E
				item_index_in_E = bisect_left(prefix_node._E,item)
				item_axis_index = prefix_node._axis_index_of_items[item_index_in_E]
				item_axis_start_index = prefix_node._axes_start_index[item_axis_index]
				items = prefix_node._items_in_axes[item_axis_index]
				item_index_in_axis = bisect_left(items,item)
				
				if extend_left:
					left_most = sort_search_continuous_left_most(items,item_index_in_axis)
					if left_most == item_index_in_axis:
						left_width = 0
					else:
						width = 0
						left_nodes = []
						for child_node_index in range(item_index_in_axis-1,left_most-1,-1):	
							suffix_node = child_nodes[item_axis_start_index+child_node_index]._search_path(suffix_items)
							if suffix_node is None:
								break
							left_nodes.append(suffix_node)
							width += 1

						left_width = min(left_width,width)
						
						extend_left_nodes[node_index].extend(left_nodes[:left_width])
						itemset_ = list(itemset)
						for i in range(left_width):
							itemset_[item_index] -= 1
							extend_left_itemsets[node_index].append(tuple(itemset_))
						
				
				#as same as extend left
				if extend_right:
					right_most = sort_search_continuous_right_most(items,item_index_in_axis,right=item_index_in_axis+right_width)
					if right_most == item_index_in_axis:
						right_width = 0
					else:
						width = 0
						right_nodes = []
						for child_node_index in range(item_index_in_axis+1,right_most+1):
							suffix_node = child_nodes[item_axis_start_index+child_node_index]._search_path(suffix_items)
							if suffix_node is None:
								break
							right_nodes.append(suffix_node)
							width += 1

						right_width = min(right_width,width)
						
						extend_right_nodes[node_index].extend(right_nodes[:right_width])
						itemset_ = list(itemset)
						for i in range(right_width):
							itemset_[item_index] += 1
							extend_right_itemsets[node_index].append(tuple(itemset_))
				
				extend_left,extend_right = left_width!=0,right_width!=0
				if not (extend_left or extend_right):
					break

			if extend_left:
				for node_index in range(len_region_nodes):
					left_nodes,left_itemsets = extend_left_nodes[node_index][:left_width],extend_left_itemsets[node_index][:left_width]

					for left_node in left_nodes:
						left_node._accessed = True
					region_nodes.extend(left_nodes)
					region_itemsets.extend(left_itemsets)
			
			if extend_right:
				for node_index in range(len_region_nodes):
					right_nodes,right_itemsets = extend_right_nodes[node_index][:right_width],extend_right_itemsets[node_index][:right_width]
					for right_node in right_nodes:
						right_node._accessed = True
					region_nodes.extend(right_nodes)
					region_itemsets.extend(right_itemsets)

			extend_left_nodes.clear()
			extend_left_itemsets.clear()
			extend_right_nodes.clear()
			extend_right_itemsets.clear()
			region_left_width.append(left_width)
			region_right_width.append(right_width)

		neighbor_nodes,neighbor_itemsets = [],[]
		for node_index in range(len(region_nodes)):
			region_itemset = region_itemsets[node_index]
			
			#check if region_nodes[node_index] is border node
			is_border = False
			for item_index in range(n_items):
				width = region_itemset[item_index]-centre_itemset[item_index]
				if width==-left_width or width==right_width:
					is_border = True
					break

			if is_border:
				neighbor_nodes_,neighbor_itemsets_ = region_nodes[node_index]._get_neighbor(root,set_accessed=False,itemset=region_itemset)
				neighbor_nodes.extend(neighbor_nodes_)
				neighbor_itemsets.extend(neighbor_itemsets_)

		return region_left_width,region_right_width,region_nodes,region_itemsets,neighbor_nodes,neighbor_itemsets

	def _get_cluster_maximal_region(self,root):
		regions_centre_itemset,regions_left_width,regions_right_width,regions_nodes,regions_itemsets = [],[],[],[],[]

		access_list = [(self,self._get_itemset())]
		while len(access_list)>0:
			node,itemset = access_list.pop()
			region_left_width,region_right_width,region_nodes,region_itemsets,neighbor_nodes,neighbor_itemsets = node._get_maximal_region(root,itemset=itemset)
			
			if not region_left_width is None:
				access_list.extend(zip(neighbor_nodes,neighbor_itemsets))
				regions_centre_itemset.append(itemset)
				regions_left_width.append(region_left_width)
				regions_right_width.append(region_right_width)
				regions_itemsets.append(region_itemsets)

		return regions_centre_itemset,regions_left_width,regions_right_width,regions_itemsets

#字典树，参照<<A Tree Projection Algorithm For Generation of Frequent Itemsets>>实现
#我们首先对数据空间进行partition，再对每一个维度的intervals递增地从0开始编码。则每一个维度的每个intervals的编码都是一个item，将所有维度的item按顺序组合后则形成itemset
#由于属于同一维度的items不可能同时出现在任一数据对象对应的itemset中，所实现的字典树区别于论文中的字典树
class trie:
	def __init__(self,data,support,min_n_objects=None,n_intervals=None,intervals_size=None,interest_measure=None,*interest_measure_args,**interest_measure_kwargs):
		self._root = trie_node(0,None,None)
		self._knots_data,self._knots_codes,self._data,counts = partition_data_space(data,n_intervals,intervals_size)
		self._support,self._min_n_objects = support,support if min_n_objects is None else min_n_objects

		if interest_measure == 'mdl':
			interest_measure = mdl_interest
		elif interest_measure == 'entropy':
			interest_measure = entropy_interest
		elif not interest_measure is None and not callable(interest_measure):
			raise ValueError("interest_measure must be str like 'mdl'、'entropy' or callable object")

		self._interest_measure,self._interest_measure_args,self._interest_measure_kwargs = interest_measure,interest_measure_args,interest_measure_kwargs

		data,root = self._data,self._root
		n_items = sum([len(knots) for knots in self._knots_data])

		for item in range(n_items):
			if counts[item]>=support:
				new_node = trie_node(1,item,root)
				new_node._count = counts[item]
				root._child_nodes.append(new_node)
				root._AE.append(item)
				root._F.append(item)
				root._E.append(item)
		
		root._gen_axes(self._knots_codes)
		self._k,self._end = (1,False) if len(root._AE)>0 else (0,True)
	
	#generate (k+1)-nodes
	def move_on(self):
		if not self._end:
			self._project_transaction()
			self._gen_nodes()
			self._subspace_pruning()
			if len(self._root._AE)==0:
				self._k -= 1
				self._end = True
	
	def get_frequent_itemsets(self,level=None):
		if not (level is None or type(level)==int):
			raise ValueError('level should be integer or None(all levels)')
		
		root = self._root
		freq_itemsets = [[] for _ in range(self._k)] if level is None else []

		access_list = root._child_nodes.copy()
		access_list.reverse()
		while len(access_list)>0:
			node = access_list.pop()
			if level is None or node._level<level:
				access_list.extend(node._child_nodes[::-1])
			
			if level is None:
				freq_itemsets[node._level-1].append(node._get_itemset())
			elif node._level==level:
				freq_itemsets.append(node._get_itemset())

		return freq_itemsets
	
	#get all level-dimension clusters,level should larger than 1
	def get_clusters(self,level,return_itemsets=True,return_nodes=False,return_count=False):
		if level<=1:
			raise ValueError('level(%u) should larger than 1'%level)
		elif level>self._k:
			raise ValueError('level(%u) should not be larger than the depth(%u) of trie'%(level,self._k))
		res = []
		if return_itemsets:
			clusters_itemsets = []
			res.append(clusters_itemsets)
		if return_nodes:
			clusters_nodes = []
			res.append(clusters_nodes)
		if return_count:
			clusters_itemsets_count = []
			res.append(clusters_itemsets_count)
		if not (return_itemsets or return_nodes):
			return None

		knots,root = self._knots_codes,self._root
		k,min_n_objects = self._k,self._min_n_objects
		start_nodes,clusters_axes = [],[]
		access_list = [root]
		while len(access_list)>0:
			node = access_list.pop()
			if node._level<level:
				if level==k:
					items = node._AE
					for child_node in node._child_nodes[::-1]:
						if sort_search(items,child_node._item) is None:
							continue
						access_list.append(child_node)
				else:
					access_list.extend(node._child_nodes[::-1])
			else:
				if not node._accessed:
					start_nodes.append(node)
					itemset = node._get_itemset()
					cluster_itemsets,cluster_nodes,cluster_itemsets_count = node._get_cluster(root,return_itemsets=return_itemsets,return_nodes=return_nodes,return_count=return_count)
					
					n_objects = sum([cluster_node._count for cluster_node in cluster_nodes])
					
					if n_objects>=min_n_objects and (return_itemsets or return_nodes):
						axes = (np.searchsorted(knots,itemset,side='right')-1).tolist()
						insert_index = bisect_left(clusters_axes,axes)
						if insert_index==len(clusters_axes) or clusters_axes[insert_index]!=axes:
							clusters_axes.insert(insert_index,axes)
							if return_itemsets:
								clusters_itemsets.insert(insert_index,[])
							if return_nodes:
								clusters_nodes.insert(insert_index,[])
							if return_count:
								clusters_itemsets_count.insert(insert_index,[])

						if return_itemsets:
							clusters_itemsets[insert_index].append(cluster_itemsets)
						if return_nodes:
							clusters_nodes[insert_index].append(cluster_nodes)
						if return_count:
							clusters_itemsets_count[insert_index].append(cluster_itemsets_count)

		for start_node in start_nodes:
			start_node._reset_accessed(root)

		return [clusters_axes]+res

	def _project_transaction(self):
		root = self._root
		data,knots = self._data,self._knots_codes
		k,support = self._k,self._support

		for T in data:
			root._projected_T.extend(sort_intersection(T,root._F))
			access_list = [root]
			while len(access_list)>0:
				node = access_list.pop()

				level,projected_T = node._level,node._projected_T

				if len(projected_T)<k-level+1:
					projected_T.clear()
					continue
				
				items,n_items = node._AE,len(node._AE)
				if level!=k-1:
					for child_node in node._child_nodes:
						item_child = child_node._item
						if sort_search(items,item_child) is None:
							continue
						
						item_index_in_projected_T = sort_search(projected_T,item_child)
						if item_index_in_projected_T is None:
							continue
						
						child_node._projected_T.extend(sort_intersection(projected_T[item_index_in_projected_T+1:],child_node._F))
						access_list.append(child_node)
				else:
					axes,axes_start_index,axis_index_of_items,items_in_axes = node._axes,node._axes_start_index,node._axis_index_of_items,node._items_in_axes
					if not hasattr(node,'_count_matrix'):
						n_axes = len(axes)
						if n_axes<=1:
							continue

						node._count_matrix = []
						start_index = axes_start_index[0]
						for axis_index in range(n_axes):
							end_index = axes_start_index[axis_index+1] if axis_index<n_axes-1 else len(items)
							node._count_matrix.append([[0]*(n_items-end_index) for _ in range(end_index-start_index)])

					count_matrix = node._count_matrix
					n_projected_T = len(projected_T)
					for item_1_index_in_T in range(n_projected_T-1):
						item_1 = projected_T[item_1_index_in_T]
						item_1_index_in_items = bisect_left(items,item_1)
						item_1_axis_index = axis_index_of_items[item_1_index_in_items]
						item_1_index_in_axis = bisect_left(items_in_axes[item_1_axis_index],item_1)

						for item_2_index_in_T in range(item_1_index_in_T+1,n_projected_T):
							item_2 = projected_T[item_2_index_in_T]
							item_2_index_in_items = bisect_left(items,item_2)
							item_2_axis_index = axis_index_of_items[item_2_index_in_items]
							if item_1_axis_index==item_2_axis_index:
								continue
							
							other_axes_start_index = axes_start_index[item_1_axis_index+1]
							item_2_index_not_in_axis = bisect_left(items[other_axes_start_index:],item_2)
							count_matrix[item_1_axis_index][item_1_index_in_axis][item_2_index_not_in_axis] += 1
				
				projected_T.clear()
	
	def _gen_nodes(self):
		root = self._root
		data,knots = self._data,self._knots_codes
		k,support = self._k,self._support

		access_list = [root]
		while len(access_list)>0:
			node = access_list.pop()
			items,n_items = node._AE,len(node._AE)
			if node._level<k-1:
				for child_node in node._child_nodes[::-1]:
					if sort_search(items,child_node._item) is None:
						continue
					access_list.append(child_node)
				node._F.clear()
				node._AE.clear()

			elif node._level==k-1:
				child_nodes = node._child_nodes
				access_list.extend(child_nodes[::-1])
				if not hasattr(node,'_count_matrix'):
					continue

				axes,axes_start_index,axis_index_of_items,items_in_axes,count_matrix = node._axes,node._axes_start_index,node._axis_index_of_items,node._items_in_axes,node._count_matrix
				for item_1_index_in_items in range(n_items-1):
					item_1 = items[item_1_index_in_items]
					item_1_axis_index = axis_index_of_items[item_1_index_in_items]
					item_1_index_in_axis = bisect_left(items_in_axes[item_1_axis_index],item_1)
					child_node = child_nodes[item_1_index_in_items]
					for item_2_index_in_items in range(item_1_index_in_items+1,n_items):
						item_2 = items[item_2_index_in_items]
						item_2_axis_index = axis_index_of_items[item_2_index_in_items]
						if item_1_axis_index==item_2_axis_index:
							continue
						
						other_axes_start_index = axes_start_index[item_1_axis_index+1]
						item_2_index_not_in_axis = bisect_left(items[other_axes_start_index:],item_2)

						if count_matrix[item_1_axis_index][item_1_index_in_axis][item_2_index_not_in_axis]>=support:
							new_node = trie_node(k+1,item_2,child_node)
							new_node._count = count_matrix[item_1_axis_index][item_1_index_in_axis][item_2_index_not_in_axis]
							child_node._child_nodes.append(new_node)
							child_node._AE.append(item_2)
							child_node._F.append(item_2)
							child_node._E.append(item_2)
					child_node._gen_axes(knots)

				del node._count_matrix
				node._F.clear()
				node._AE.clear()
		
			else:
				if len(node._child_nodes)>0:		#active node
					parent_node = node._parent_node
					while not parent_node is None:
						if len(parent_node._AE)==0 or parent_node._AE[-1]!=node._item:
							parent_node._AE.append(node._item)
						sort_union(parent_node._F,[node._item]+node._F,inplace=True)
						node,parent_node = parent_node,parent_node._parent_node

		self._k += 1
	
	def _subspace_pruning(self):
		if self._interest_measure is None:
			return
		
		clusters_axes,clusters_nodes,clusters_itemsets_count = self.get_clusters(self._k,return_itemsets=False,return_nodes=True,return_count=True)
		pop_indices = self._interest_measure(clusters_itemsets_count,*self._interest_measure_args,**self._interest_measure_kwargs)
		for pop_index in pop_indices:
			for cluster_nodes in clusters_nodes[pop_index]:
				for node in cluster_nodes:
					node._delete_node(itemset_axes=clusters_axes[pop_index])

def is_belong_to_cluster(codes,subspace,regions_centre_itemset,regions_left_width,regions_right_width):
	n_regions = len(regions_centre_itemset)
	belong_to_cluster = np.zeros(len(codes),np.bool8)
	
	for sample_index in range(len(codes)):
		itemset = codes[sample_index][subspace]
		dist = itemset - regions_centre_itemset
		in_left_region,in_right_region = np.logical_and(dist<=0,-dist<=regions_left_width),np.logical_and(dist>=0,dist<=regions_right_width)
		in_region = np.logical_or(in_left_region,in_right_region)
		
		in_cluster = np.nonzero(np.all(in_region,axis=1))[0]
		if in_cluster.shape[0]>0:
			belong_to_cluster[sample_index] = True
	
	return belong_to_cluster

def predict_clique(data,subspace,clusters_centre_itemsets,clusters_left_width,clusters_right_width,knots_data,knots_codes,do_partition=True):
	if do_partition:
		codes = np.empty(data.shape,np.uint32)
		for attribute_index in range(data.shape[1]):
			codes[:,attribute_index] = np.searchsorted(knots_data[attribute_index],data[:,attribute_index],side='right')-1 + knots_codes[attribute_index]
		data = codes
	
	cluster_ids = np.empty(data.shape[0],np.int32)
	cluster_ids[:] = -1
	for cluster_index in range(len(clusters_centre_itemsets)):
		belong_to_cluster_indices = is_belong_to_cluster(data,subspace,clusters_centre_itemsets[cluster_index],clusters_left_width[cluster_index],clusters_right_width[cluster_index])
		cluster_ids[belong_to_cluster_indices] = cluster_index
	return cluster_ids	

def clique(data,k,support,n_intervals,min_n_objects=None,return_predict_info=False,interest_measure=None,*interest_measure_args,**interest_measure_kwargs):
	"""
		CLIQUE算法，参照论文<<Automatic Subspace Clustering of High Dimensional Data for Data Mining Applications>>
		参数：
			①data：2D array，存储数据对象
			②k：整型，表示进行聚类的子空间的最高维度。
			③support：整型，density threshold γ，表示dense unit包含的最少数据对象
			④n_intervals：整型，参数ξ
			⑤min_n_object：整型，表示构成cluster的最小数据对象数目。默认为None，值与support相同
			⑥return_predict_info：bool，是否返回用于预测的信息。默认为False
			⑦interest_measure：str或callable object，用于对子空间进行剪枝，子空间剪枝可以提高算法速度。若为str，则可选为'mdl'或'entropy'，表示使用minimal description length或entropy作为剪枝标准。若为callable object，其第一个参数应为subspaces_itemsets_count，该参数为一list of lists of integer，即各个子空间的各个簇的各个dense units包含的数据对象个数。函数应返回一个list，表示要被移除的子空间的index。interest_measure默认为None，即表示不对子空间进行剪枝
			⑧*interest_measure_args,**interest_measure_kwargs：传入interest_measure中的参数。若interest_measure为'mdl'或'entropy'，则参数有least_keep_ratio，表示至少保留的子空间的比例
	"""
	tree = trie(data,support,min_n_objects,n_intervals,None,interest_measure,*interest_measure_args,**interest_measure_kwargs)

	pre_tree_k = 0
	while True:
		tree.move_on()
		if tree._k == k or tree._k == pre_tree_k:
			break
		pre_tree_k = tree._k
	
	root,data = tree._root,tree._data

	subspaces,subspaces_cluster_nodes = tree.get_clusters(tree._k,return_itemsets=False,return_nodes=True,return_count=False)
	if len(subspaces)==0:
		return None
	else:
	#if there are at least 1 k-dimension subspace,choose the "bestSubspace" according to <<Density-Connected Subspace Clustering for High-Dimensional Data>>
		min_subspace_n_cluster_object,min_subspace,min_subspace_cluster_node = np.inf,None,None
		for subspace_index in range(len(subspaces)):
			n_cluster_object = sum([len(cluster_nodes) for cluster_nodes in subspaces_cluster_nodes[subspace_index]])
			if n_cluster_object<min_subspace_n_cluster_object:
				min_subspace_n_cluster_object,min_subspace = n_cluster_object,subspaces[subspace_index]
				min_subspace_cluster_node = [cluster_nodes[0] for cluster_nodes in subspaces_cluster_nodes[subspace_index]]
			
			subspaces_cluster_nodes[subspace_index].clear()
	
		cluster_nodes = min_subspace_cluster_node
		clusters_centre_itemsets,clusters_left_width,clusters_right_width,clusters_itemsets = [],[],[],[]
		for cluster_index in range(len(cluster_nodes)):
			regions_centre_itemset,regions_left_width,regions_right_width,regions_itemsets = cluster_nodes[cluster_index]._get_cluster_maximal_region(root)
			
			clusters_centre_itemsets.append(regions_centre_itemset)
			clusters_left_width.append(regions_left_width)
			clusters_right_width.append(regions_right_width)
			clusters_itemsets.append(regions_itemsets)
	
		pop_indices = remove_redundant_regions(clusters_itemsets)
		for pop_index,index in zip(pop_indices,range(len(pop_indices))):
			pop_index -= index
			clusters_centre_itemsets.pop(pop_index)
			clusters_left_width.pop(pop_index)
			clusters_right_width.pop(pop_index)
		del clusters_itemsets

		predict_info = (min_subspace,clusters_centre_itemsets,clusters_left_width,clusters_right_width,tree._knots_data,tree._knots_codes)
		cluster_ids = predict_clique(data,*predict_info,do_partition=False)

		return (cluster_ids,predict_info) if return_predict_info else cluster_ids

