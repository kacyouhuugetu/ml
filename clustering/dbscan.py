import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors.dist_metrics import EuclideanDistance

_UNCLASSIFIED_,_NOISE_ = -2,-1

def expand_cluster(X,index,minpts,rad,cluster_ids,current_cluster_id,kdtree,return_predict_info,core_point_indices,core_point_cluster_ids):
	neighbor_indices,neighbor_dists = kdtree.query_radius(X[index,np.newaxis],rad,return_distance=True,sort_results=True)
	neighbor_indices,neighbor_dists = neighbor_indices[0],neighbor_dists[0]
	
	#border or noise point
	if neighbor_indices.shape[0]<minpts:
		cluster_ids[index] = _NOISE_		#noise
		return False
	#core point
	if neighbor_indices.shape[0]>=minpts:
		if return_predict_info:
			core_point_indices.append(index)
			core_point_cluster_ids.append(current_cluster_id)
		cluster_ids[neighbor_indices] = current_cluster_id

		seeds = list(neighbor_indices[1:])
		while len(seeds)>0:
			current_object_index = seeds.pop()
			
			neighbor_indices,neighbor_dists = kdtree.query_radius(X[current_object_index,np.newaxis],rad,return_distance=True,sort_results=True)
			neighbor_indices,neighbor_dists = neighbor_indices[0],neighbor_dists[0]
			
			if neighbor_indices.shape[0]>=minpts:
				if return_predict_info:
					core_point_indices.append(current_object_index)
					core_point_cluster_ids.append(current_cluster_id)
				for neighbor_index in neighbor_indices:
					#NOISE(-1) or UNCLASSIFIED(-2)
					if cluster_ids[neighbor_index]<0:
						#UNCLASSIFIED,so X[neighbor_index] may be a core point
						if cluster_ids[neighbor_index] == _UNCLASSIFIED_:
							seeds.append(neighbor_index)
						cluster_ids[neighbor_index] = current_cluster_id
		return True

def dbscan(X,minpts,rad,metric=None,return_predict_info=False):
	"""
		DBSCAN算法
		参数：
			①X：2D array。存储数据对象
			②minpts、rad：OPTICS算法参数，rad-neighborhood的大小大于等于minpts的数据对象将成为core-point
			③metric：用于计算两个数据对象距离的对象，应使用sklearn.neighbors.dist_metrics中的距离对象。若metric为None，则使用EuclideanDistance
	"""
	metric = EuclideanDistance() if metric is None else metric
	N = X.shape[0]
	kdtree = KDTree(X,metric=metric)
	_indices_ = np.random.choice(range(N),replace=False,size=N)
	
	cluster_ids,current_cluster_id = np.empty(N,np.int32),0
	cluster_ids[:] = _UNCLASSIFIED_		#initialize all element to UNCLASSIFIED
	if return_predict_info:
		core_point_indices,core_point_cluster_ids = [],[]
	else:
		core_point_indices,core_point_cluster_ids = None,None

	for index in _indices_:
		#UNCLASSIFIED
		if cluster_ids[index] == _UNCLASSIFIED_:
			current_cluster_id += expand_cluster(X,index,minpts,rad,cluster_ids,current_cluster_id,kdtree,return_predict_info,core_point_indices,core_point_cluster_ids)

	if return_predict_info:
		core_point_indices,core_point_cluster_ids = np.array(core_point_indices,dtype=np.uint32),np.array(core_point_cluster_ids,dtype=np.uint32)
		sorted_indices = np.argsort(core_point_indices)
		predict_info = (rad,kdtree,core_point_indices[sorted_indices],core_point_cluster_ids[sorted_indices])

	return (cluster_ids,predict_info) if return_predict_info else cluster_ids

#一个简单地实现预测聚类的方法
#算法迭代每一个样本来判断其rad-近邻是否含有core point，若有，则该样本属于该core point所属的cluster，否则为noise
def predict_dbscan(X,rad,kdtree,core_point_indices,core_point_cluster_ids):
	cluster_ids = np.empty(X.shape[0],np.int32)
	cluster_ids[:] = _NOISE_
	n_core_points = len(core_point_indices)

	for index in range(X.shape[0]):
		neighbor_indices = kdtree.query_radius(X[index,np.newaxis],rad)[0]
		indices = np.searchsorted(core_point_indices,neighbor_indices,side='left')
		indices[indices==n_core_points] = -1
		nz_index = np.nonzero(core_point_indices[indices]==neighbor_indices)[0]

		if len(nz_index)>0:
			nz_index = nz_index[0]
			index_in_core_point_indices = np.searchsorted(core_point_indices,neighbor_indices[nz_index])
			cluster_ids[index] = core_point_cluster_ids[index_in_core_point_indices]
	
	return cluster_ids
