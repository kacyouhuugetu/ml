import numpy as np
from random import choice
from sklearn.neighbors.dist_metrics import EuclideanDistance
from .cluster_validation import dunn_index,DB_index,average_dissimilarity

def change_cluster_medoids(X,changed_cluster_id,neighbor_index,cluster_ids,medoids,dists_to_medoids,metric,_indices_=None):
	medoids,cluster_ids,dists_to_medoids = medoids.copy(),cluster_ids.copy(),dists_to_medoids.copy()
	_indices_ = np.arange(X.shape[0],dtype=np.uint32) if _indices_ is None else _indices_
	medoids[changed_cluster_id],cluster_ids[neighbor_index] = X[neighbor_index],changed_cluster_id
	
	id_indices = cluster_ids==changed_cluster_id
	dists = metric.pairwise(X[id_indices],medoids)
	
	new_cluster_ids = np.argmin(dists,axis=1)
	dists = dists[_indices_[:new_cluster_ids.shape[0]],new_cluster_ids]
	cluster_ids[id_indices] = new_cluster_ids
	dists_to_medoids[id_indices] = dists
	
	not_id_indices = np.logical_not(id_indices,out=id_indices)
	x = X[not_id_indices]
	if x.shape[0]==0:
		return medoids,cluster_ids,dists_to_medoids

	dists = metric.pairwise(X[neighbor_index,np.newaxis],x)[0]
	
	changed_indices = dists<dists_to_medoids[not_id_indices]
	changed_indices,dists = _indices_[not_id_indices][changed_indices],dists[changed_indices]
	cluster_ids[changed_indices] = changed_cluster_id
	dists_to_medoids[changed_indices] = dists

	return medoids,cluster_ids,dists_to_medoids

def clarans(X,K,n_local=1,max_n_neighbor=-1,metric=None,cost_calculator='average dissimilarity',buf_size=None,buf_ratio=1.,return_predict_info=False):
	"""
		聚类算法CLARANS
		参数：
			①X：2D array。存储数据对象
			②K：簇的个数
			③metric：用于计算两个数据对象距离的对象，应使用sklearn.neighbors.dist_metrics中的距离对象。若metric为None，则使用EuclideanDistance
			④n_local：local minimal cost的寻找次数。默认为1
			⑤max_n_neighbor：访问的邻接结点的最大个数，若max_n_neighbor<=0则表示访问所有邻接结点，默认为-1
			⑥cost_calculator：str。表示用于计算聚类质量的函数。
				1)当cost_calculator为'average dissimilarity'时，表示用各个簇的dissimilarity均值
				2)当cost_calculator为'dunn'时，表示使用dunn index
				3)当cost_calculator为'DB'时，表示使用Davies–Bouldin index
				4)否则，cost_calculator为callable object，其接受参数X表示数据，参数K表示簇的个数，参数cluster_ids表示各个数据对象所属的簇，参数medoids表示各个簇的质心，其与参数由cost_calculator_args指定
			⑦buf_size,buf_ratio：计算簇的各个数据对象之间距离的批处理量
	"""

	if cost_calculator=='average dissimilarity':
		cost_calculator = average_dissimilarity
	elif cost_calculator=='dunn':
		cost_calculator = lambda *_,**__:-dunn_index(*_,**__)
	elif cost_calculator=='DB':
		cost_calculator = DB_index
	else:
		raise ValueError("cost_calculator must be str like 'average dissimilarity','dunn' or 'DB'")

	N = X.shape[0]
	max_n_neighbor = N if max_n_neighbor<=0 else max_n_neighbor
	metric = EuclideanDistance() if metric is None else metric
	_indices_ = np.arange(N,dtype=np.uint32)
	
	min_cost,min_cluster_ids = np.inf,None
	for repeat_index in range(n_local):
		medoids = X[np.random.choice(_indices_,replace=False,size=K)]
		dists_to_medoids = metric.pairwise(X,medoids)
		cluster_ids = np.argmin(dists_to_medoids,axis=1)
		dists_to_medoids = dists_to_medoids[_indices_,cluster_ids]
		
		cost = cost_calculator(X,K,cluster_ids,medoids,None,metric,buf_size,buf_ratio)

		n_neighbor = 0
		while True:
			if n_neighbor==max_n_neighbor:
				break

			while True:
				neighbor_index = choice(_indices_)
				if not np.any([np.allclose(medoid,X[neighbor_index]) for medoid in medoids]):
					break
			
			changed_cluster_id = choice(range(K))
			medoids_,cluster_ids_,dists_to_medoids_ = change_cluster_medoids(X,changed_cluster_id,neighbor_index,cluster_ids,medoids,dists_to_medoids,metric,_indices_)
			cost_ = cost_calculator(X,K,cluster_ids_,medoids,None,metric,buf_size,buf_ratio)

			if cost_<cost:
				cost = cost_
				medoids,cluster_ids,dists_to_medoids = medoids_,cluster_ids_,dists_to_medoids_
				n_neighbor = 0
			else:
				n_neighbor += 1

		if cost<min_cost:
			min_cost,min_medoids,min_cluster_ids = cost,medoids,cluster_ids

	return (min_cluster_ids,(min_medoids,metric)) if return_predict_info else min_cluster_ids

def predict_clarans(X,medoids,metric):
	cluster_ids = np.empty(X.shape[0],np.int32)
	
	for index in range(X.shape[0]):
		dists = metric.pairwise(X[index,np.newaxis],medoids)[0]
		cluster_ids[index] = np.argmin(dists)

	return cluster_ids
