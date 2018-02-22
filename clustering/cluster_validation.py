import numpy as np
from sklearn.neighbors.dist_metrics import EuclideanDistance

def calculate_cluster_distance(c1,c2,use_medoids_dists=False,single_link=True,complete_link=False,dists=None,metric=None,buf_ratio=1.):
	"""
		计算两个簇的"距离"
		参数：
			①c1、c2：两个簇。若use_to_medois_dists为True，则c1、c2为分别为各个簇的质心，否则为2D array，表示各个簇的数据对象
			②use_medoids_dists：是否根据簇的质心来计算簇的距离，默认为False
			③single_link、complete_link：如何计算簇的距离。
				1)当single_link为True(默认)，则利用两个簇最相近的两个数据对象的距离作为两个簇的距离；
				2)当single_link为False而complete_link为True时，则利用两个簇最相远的两个数据对象的距离作为两个簇的距离；
				3)当single_link为False且complete_link为False时，则使用average link，即将两个簇各个数据对象之间的平均距离作为两个簇的距离
			④dists：2D array，表示两个簇各个数据对象之间的距离。默认为None
			⑤metric：用于计算两个数据对象距离的对象，应使用sklearn.neighbors.dist_metrics中的距离对象。若metric为None，则使用EuclideanDistance
			⑥buf_ratio：计算簇的各个数据对象之间距离的批处理量
	"""
	if use_medoids_dists:
		metric = EuclideanDistance() if metric is None else metric
		return metric.pairwise(c1[np.newaxis,:],c2[np.newaxis,:])[0,0]
	
	if c1.shape[0]==0 or c2.shape[0]==0:
		return np.inf

	if dists is None:
		metric = EuclideanDistance() if metric is None else metric
		
		n_objects = c1.shape[0]
		buf_size = min(n_object,max(1,int(n_objects*buf_ratio)))
		distance = np.inf if single_link else -np.inf if complete_link else 0.
		start_index,end_index = 0,buf_size
		while start_index<n_objects:
			dists = metric.pairwise(c1[start_index:end_index],c2)
			start_index,end_index = end_index,end_index+buf_size
			
			distance = min(distance,np.min(dists)) if single_link else \
						max(distance,np.max(dists)) if complete_link else \
						distance + np.sum(dists)
		if not (single_link or complete_link):
			distance /= c1.shape[0]*c2.shape[0]
	else:
		distance = np.min(dists) if single_link else \
					np.max(dists) if complete_link else \
					np.min(dists)
	
	return distance

def calculate_cluster_dispersion(c,medoid=None,use_to_medoids_dists=False,dists=None,metric=None,buf_ratio=1.):
	"""
		计算簇的"散布"
		参数：
			①c：2D array，表示簇内的数据对象
			②medoid：2D array，表示簇的质心，默认为None。
			③use_to_medoids_dists：默认为False。若use_to_medoids_dists为True，则利用簇内各个数据对象到簇质心的平均距离作为簇的"散布"，否则将使用簇中两个数据对象的最大距离作为簇的"散布"
			④dists：1D/2D array，若use_to_medoids_dists为True，则表示簇的各个数据对象到簇质心的距离，否则表示两个簇各个数据对象之间的距离。默认为None
			⑤metric：用于计算两个数据对象距离的对象，应使用sklearn.neighbors.dist_metrics中的距离对象。若metric为None，则使用EuclideanDistance
			⑥buf_ratio：计算簇的各个数据对象之间距离的批处理量
	"""
	if dists is None:
		metric = EuclideanDistance() if metric is None else metric

		n_objects = c.shape[0]
		buf_size = min(n_object,max(1,int(n_objects*buf_ratio)))
		
		distance = 0. if use_to_medoids_dists else -np.inf 
		start_index,end_index = 0,buf_size
		while start_index<n_objects:
			dists = metric.pairwise(c[start_index:end_index],medoid[np.newaxis] if use_to_medoids_dists else c)
			start_index,end_index = end_index,end_index+buf_size
			
			distance = distance + np.sum(dists) if use_to_medoids_dists else max(distance,np.max(dists))
		if use_to_medoids_dists:
			distance /= c.shape[0]
	else:
		distance = np.sum(dists)/c.shape[0] if use_to_medoids_dists else np.max(dists)
	
	return distance

def get_cluster_dispersion_and_distance(X,K,cluster_ids,medoids=None,dists=None,use_to_medoids_dists=False,use_medoids_dists=False,single_link=True,complete_link=False,metric=None,buf_ratio=1.):
	"""
		计算各个簇之间的"距离"和各个簇的"散布"
		参数：
			①X：2D array，表示数据对象
			②K：簇的数量
			③cluster_ids：1D array，表示各个数据对象所属的簇，取值必须从0到K-1
			④medoids：2D array，表示簇的质心，默认为None。
			⑤dists：1D/2D array，若use_medoids_dists为True，则表示簇的各个数据对象到簇质心的距离，否则表示两个簇各个数据对象之间的距离。默认为None
			⑥use_medoids_dists：是否根据簇的质心来计算簇的距离，默认为False
			⑦use_to_medoids_dists：默认为False。若use_to_medoids_dists为True，则利用簇内各个数据对象到簇质心的平均距离作为簇的"散布"，否则将使用簇中两个数据对象的最大距离作为簇的"散布"
			⑧single_link、complete_link：如何计算簇的距离。
				1)当single_link为True(默认)，则利用两个簇最相近的两个数据对象的距离作为两个簇的距离；
				2)当single_link为False而complete_link为True时，则利用两个簇最相远的两个数据对象的距离作为两个簇的距离；
				3)当single_link为False且complete_link为False时，则使用average link，即将两个簇各个数据对象之间的平均距离作为两个簇的距离
			⑨metric：用于计算两个数据对象距离的对象，应使用sklearn.neighbors.dist_metrics中的距离对象。若metric为None，则使用EuclideanDistance
			⑩buf_ratio：计算簇的各个数据对象之间距离的批处理量
	"""

	dispersions = np.empty(K,np.float64)
	distances = np.empty((K,K),np.float64)
	
	for cluster_id in range(K):
		c1_indices = cluster_ids==cluster_id
		c1,medoid1 = X[c1_indices],medoids[cluster_id] if use_to_medoids_dists or use_medoids_dists else None

		if not dists is None:
			c1_dists = dists[c1_indices]
			c1_dists_c1 = c1_dists if use_to_medoids_dists else c1_dists[:,c1_indices]
		else:
			c1_dists,c1_dists_c1 = None,None
		
		dispersions[cluster_id] = calculate_cluster_dispersion(c1,medoid1,use_to_medoids_dists,c1_dists_c1,metric,buf_ratio)
		
		for another_cluster_id in range(cluster_id+1,K):
			if use_medoids_dists:
				medoid2 = medoids[another_cluster_id]
				distances[cluster_id,another_cluster_id] = calculate_cluster_distance(medoid1,medoid2,use_medoids_dists,metric=metric)
			else:
				c2_indices = cluster_ids==another_cluster_id
				c2 = X[c2_indices]
				c1_dists_c2 = None if use_to_medoids_dists or c1_dists is None else c1_dists[:,c2_indices]
				distances[cluster_id,another_cluster_id] = calculate_cluster_distance(c1,c2,use_medoids_dists,single_link,complete_link,c1_dists_c2,metric,buf_ratio)

	distances[np.diag_indices(K)] = 1.		#should not use the diagonal of distance
	distances[np.tril_indices(K,-1)] = distances.T[np.tril_indices(K,-1)]

	return dispersions,distances

def dunn_index(X,K,cluster_ids,medoids=None,dists=None,use_to_medoids_dists=False,use_medoids_dists=False,single_link=True,complete_link=False,metric=None,buf_ratio=1.):
	"""
		计算Dunn index。计算公式为min(cluster distances)/max(cluster dispersions)。通常Dunn index值越大，聚类算法越好
		参数说明见函数get_cluster_dispersion_and_distance
	"""
	dispersions,distances = get_cluster_dispersion_and_distance(X,K,cluster_ids,medoids,dists,use_to_medoids_dists,use_medoids_dists,single_link,complete_link,metric,buf_ratio)
	
	return np.min(distances[np.triu_indices(K,1)])/np.max(dispersions)

def DB_index(X,K,cluster_ids,medoids=None,dists=None,use_to_medoids_dists=False,use_medoids_dists=False,single_link=True,complete_link=False,metric=None,buf_ratio=1.):
	"""
		计算Davies–Bouldin index。通常Davies–Bouldin index值越小，聚类算法越好
		参数说明见函数get_cluster_dispersion_and_distance
	"""
	dispersions,distances = get_cluster_dispersion_and_distance(X,K,cluster_ids,medoids,dists,use_to_medoids_dists,use_medoids_dists,single_link,complete_link,metric,buf_ratio)

	R = ( dispersions + dispersions[:,np.newaxis] ) / distances
	R[np.diag_indices(K)] = -np.inf

	return np.sum(np.max(R,axis=1))/K

def average_dissimilarity(X,K,cluster_ids,medoids,dists_to_medoids,use_to_medoids_dists=False,use_medoids_dists=False,single_link=True,complete_link=False,metric=None,buf_ratio=1.):
	if dists_to_medoids is None:
		metric = EuclideanDistance() if metric is None else metric
		dissimilarity = 0.
		for cluster_id in range(K):
			x = X[cluster_ids==cluster_id]
			if x.shape[0] == 0:
				continue
			dissimilarity += np.sum(metric.pairwise(x,medoids[cluster_id,np.newaxis]))
		return dissimilarity/X.shape[0]

	else:
		return np.mean(dists_to_medoids)
