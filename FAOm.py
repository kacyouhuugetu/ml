from math import ceil,sqrt
from bisect import bisect_left
from .base import sort_search,sort_search_left_most,sort_search_right_most,partition_data_space
from sklearn.neighbors.dist_metrics import EuclideanDistance
import numpy as np

#寻找outlier

#_GLAY_:all points in cell are outlier
#_WHITE_:some points in cell are outlier
#_PINK_,_RED_:all points in cell are non-outlier
_GLAY_,_WHITE_,_PINK_,_RED_ = -1,0,1,2

class map_node:
	def __init__(self,item,parent_node):
		self._item,self._parent_node = None if item is None else int(item),parent_node
		self._child_nodes,self._child_items = [],[]

	def _get_itemset(self):
		itemset = []
		node,parent_node = self,self._parent_node
		while not parent_node is None:
			itemset.append(node._item)
			node,parent_node = parent_node,parent_node._parent_node

		itemset.reverse()
		return itemset

	def _search_path(self,itemset):
		node = self
		for item in itemset:
			item_index_in_items = sort_search(node._child_items,item)
			if item_index_in_items is None:
				return None
			node = node._child_nodes[item_index_in_items]
		return node

class map_tree:
	def __init__(self,X,p,D):
		self._root = map_node(None,None)
		self._X,self._p,self._D,self._M,self._cell_width = X,p,D,int(X.shape[0]*(1-p)),ceil(2*sqrt(X.shape[1]))
		self._knots_data,self._knots_codes,codes,_ = partition_data_space(X,intervals_size=self._cell_width)

		M = self._M
		root = self._root
		n_code = len(codes[0])
		reds = []

		for sample_index in range(X.shape[0]):
			node,x,code = root,X[sample_index],codes[sample_index]
			for index in range(n_code):
				item,items = code[index],node._child_items
				item_index_in_items = bisect_left(items,item)
				if item_index_in_items==len(items) or items[item_index_in_items]!=item:
					new_node = map_node(item,node)
					if index==n_code-1:
						new_node._color = _WHITE_
						new_node._count,new_node._sample_indices = 0,[]

					items.insert(item_index_in_items,item)
					node._child_nodes.insert(item_index_in_items,new_node)

				node = node._child_nodes[item_index_in_items]
			
			if node._count==M:
				node._color = _RED_
				reds.append(node)
			node._count += 1
			node._sample_indices.append(sample_index)

		for red_node in reds:
			itemset = red_node._get_itemset()
			for L1_node in self._get_neighbors(itemset,1):
				if L1_node._color == _WHITE_:
					L1_node._color = _PINK_

	def _get_neighbors(self,code,outer_width,inner_width=None):
		root = self._root
		access_list = [(0,root)]
		neighbor_nodes = []
		
		exclude_inner = False if inner_width is None else True
		while len(access_list)>0:
			level,node = access_list.pop()
			child_nodes = node._child_nodes
			if len(child_nodes)>0:
				items,item = node._child_items,code[level]
				left_most,right_most = sort_search_left_most(items,item-outer_width),sort_search_right_most(items,item+outer_width)
				if left_most == len(items) or right_most==-1:
					continue
				
				for index in range(left_most,right_most+1):
					if exclude_inner and abs(child_nodes[index]._item-item)<=inner_width:
						continue
					access_list.append((level+1,child_nodes[index]))
			else:
				neighbor_nodes.append(node)

		return neighbor_nodes

	def findallout(self):
		X,D,M = self._X,self._D,self._M
		ed = EuclideanDistance()
		cell_width = self._cell_width
		
		outlier_indices = []
		access_list = [self._root]
		while len(access_list)>0:
			node = access_list.pop()
			if len(node._child_nodes)>0:
				access_list.extend(node._child_nodes)

			elif node._color == _WHITE_:
				count_L1,count_L2 = 0,0
				itemset,sample_indices = node._get_itemset(),node._sample_indices
				for L1_node in self._get_neighbors(itemset,outer_width=1):
					count_L1 += L1_node._count
				if count_L1>M:
					node._color == _PINK_
					continue

				L2_nodes = self._get_neighbors(itemset,outer_width=cell_width,inner_width=1)
				for L2_node in L2_nodes:
					count_L2 += L2_node._count
				if count_L2<=M:
					node._color = _GLAY_
					outlier_indices.extend(sample_indices)
					continue

				for index in range(count.shape[0]):
					count = count_L1
					for L2_node in L2_nodes:
						dists = ed.pairwise(X[sample_indices],X[L2_node._sample_indices])
						count[index] += np.sum(dists<=D)
						if count[index]>M:
							break
					else:
						outlier_indices.append(sample_indices[index])

		return outlier_indices

def FindAllOutM(X,p,D):
	tree = map_tree(X,p,D)
	return tree.findallout()
