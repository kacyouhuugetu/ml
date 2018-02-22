import numpy as np
from ..base import less_equal
from bisect import bisect_left,bisect_right
from bintrees import RBTree
from sklearn.neighbors import KDTree
from sklearn.neighbors.dist_metrics import EuclideanDistance

OPTICS_UNDEFINED = np.inf
EPS = np.finfo(float).eps

check_uppoint = lambda r_current,r_next,xi:False if r_current is OPTICS_UNDEFINED else less_equal(r_current,r_next*(1-xi),tol=1e-10)
check_downpoint = lambda r_current,r_next,xi:False if r_current is OPTICS_UNDEFINED else less_equal(r_next,r_current*(1-xi),tol=1e-10)

#check if index_start is the start of a up area or down area
def is_start_of_area(r,xi,index_start,minpts,minlen,up=True):
	if r[index_start] == OPTICS_UNDEFINED:
		return False,None
	check_point = check_uppoint if up else check_downpoint

	r_current,start_point,last_point,count_exception = r[index_start],None,None,0
	for point in range(index_start,len(r)-1):
		r_next = r[point+1]
		if check_point(r_current,r_next,xi):
			count_exception,last_point = 0,point
			if start_point is None:
				start_point = point
		
		elif start_point is None:
			return False,None
		
		else:
			count_exception += 1
			
			if ( r_next<r_current if up else r_next>r_current ) or count_exception == minpts:
				return (True,last_point) if last_point-start_point+1>=minlen else (False,None)
		
		r_current = r_next
	
	return (True,last_point) if last_point-start_point+1>=minlen else (False,None)

#delete down areas and update local mib
def update_mib_and_filter(down_area_starts,down_area_ends,down_area_mibs,xi,mib):
	n_delete = 0
	for index in range(len(down_area_mibs)):
		index_ = index-n_delete
		down_area_mibs[index_] = max(mib,down_area_mibs[index_])
		if down_area_mibs[index_]*(1-xi)<mib:
			down_area_starts.pop(index_)
			down_area_ends.pop(index_)
			down_area_mibs.pop(index_)
			n_delete += 1

#construct ξ-area by combining down area and up area
def combine_area(r,down_area_start,down_area_end,up_area_start,up_area_end,down_area_mib,xi,minpts):
	r_sd,r_eu_plus_one = r[down_area_start],r[up_area_end+1]
	
	#satisify (sc2*) and (3a)
	if r[up_area_end]*(1-xi)>=down_area_mib and up_area_end-down_area_start+1>=minpts:
		s,e = down_area_start,up_area_end

		if r_sd*(1-xi) >= r_eu_plus_one:
			r_end = len(r)-1
			s = r_end - bisect_right(r[::-1],r_eu_plus_one,r_end-down_area_end,r_end-down_area_start)
		elif r_eu_plus_one*(1-xi) >= r_sd:
			e = bisect_right(r,r_sd,up_area_start,up_area_end)
	else:
		s,e = None,None

	return s,e

#extract clusters from r plot
#implement Algorithm ExtractClusters in <<OPTICS: Ordering Points To Identify the Clustering Structure>> Figure 19
#minlen是down area或up area的最小长度
def get_clusters_auto(r,xi,minpts,minlen=1):
	mib,index = 0,0
	cluster_starts,cluster_ends = [],[]
	down_area_starts,down_area_ends,down_area_mibs = [],[],[]
	while index<len(r)-1:
		mib = max(mib,r[index])
		is_start_of_down_area,end_index = is_start_of_area(r,xi,index,minpts,minlen,up=False)
		
		if is_start_of_down_area:
			update_mib_and_filter(down_area_starts,down_area_ends,down_area_mibs,xi,mib)
			down_area_starts.append(index)
			down_area_ends.append(end_index)
			down_area_mibs.append(0.)

			index,mib = end_index + 1,0
		else:
			is_start_of_up_area,end_index = is_start_of_area(r,xi,index,minpts,minlen,up=True)
			
			if is_start_of_up_area:
				update_mib_and_filter(down_area_starts,down_area_ends,down_area_mibs,xi,mib)
				up_area_start,up_area_end = index,end_index
				index,mib = end_index +1,0

				for down_area_start,down_area_end,down_area_mib in zip(down_area_starts,down_area_ends,down_area_mibs):
					s,e = combine_area(r,down_area_start,down_area_end,up_area_start,up_area_end,down_area_mib,xi,minpts)
					if not s is None:
						cluster_starts.append(s)
						cluster_ends.append(e)

			else:
				index += 1
	
	return cluster_starts,cluster_ends

#a stupid way to get minpts different neighbors and k-distance
def get_minpts_neighbors(x,neighbor_indices,neighbor_dists,minpts,kdtree):
	search_k = max(2*neighbor_indices.shape[0],minpts)
	while True:
		ele,indices = np.unique(neighbor_dists,return_index=True,sorted=True)
		#less than minpts different neighbors
		if ele.shape[0]<minpts:
			try:
				#search neighbors with double rad
				neighbor_dists,neighbor_indices = kdtree.query(x,search_k,return_distance=True,sort_results=True)
				neighbor_dists,neighbor_indices = neighbor_dists[0],neighbor_indices[0]

			#can't find minpts different neighbors
			except ValueError:
				raise ValueError("error:can't find minpts different neighbors")

			search_k *= 2
			continue
		break

	end_index = neighbor_indices.shape[0] if ele.shape[0]==minpts else indices[minpts]
	k_distance,neighbor_indices = neighbor_dists[end_index-1],neighbor_indices[:end_index].astype(np.uint32)
	return neighbor_indices,k_distance

#获取各个数据对象的local outlier factor
def get_of(X,minpts,metric,neighbor_indices=None,k_distance=None):
	N = X.shape[0]
	kdtree = KDTree(X,metric=metric)
	lrd = np.empty(N,np.float64)
	
	if neighbor_indices is None:
		cd,neighbor_indices = np.empty(N,np.float64),np.empty(N,np.object)
		for index in range(N):
			object_neighbor_dists,object_neighbor_indices = kdtree.query(X[index,np.newaxis],minpts,return_distance=True,sort_results=True)
			object_neighbor_dists,object_neighbor_indices = object_neighbor_dists[0],object_neighbor_indices[0]
			
			neighbor_indices[index],cd[index] = get_minpts_neighbors(X,object_neighbor_indices,object_neighbor_dists,minpts,kdtree)
	else:
		cd = k_distance
	
	for index in range(N):
		neighbors = neighbor_indices[index]
		neighbor_cds,neighbor_dists = cd[neighbors],metric.pairwise(X[index,np.newaxis],X[neighbors])[0]
		rd = np.where(neighbor_cds>neighbor_dists,neighbor_cds,neighbor_dists)
		lrd[index] = neighbors.shape[0]/np.sum(rd)
	
	of = cd
	for index in range(N):
		neighbor_lrds = lrd[neighbor_indices[index]]
		of[index] = np.sum(neighbor_lrds)/lrd[index]/neighbor_lrds.shape[0]
	
	return of

#implement Method OrderSeeds::update in <<OPTICS: Ordering Points To Identify the Clustering Structure>> Figure 7
def orderseed_update(X,orderseeds,cd,rd,processed,object_index,neighbor_indices,neighbor_dists):
	cd_object = cd[object_index]
	for neighbor_index,neighbor_dist in zip(neighbor_indices,neighbor_dists):
		#update orderseeds and reach distance of neighbors
		if not processed[neighbor_index]:
			rd_neighbor = max(cd_object,neighbor_dist)
			if rd_neighbor < rd[neighbor_index]:
				if rd[neighbor_index]!=OPTICS_UNDEFINED:
					indices = orderseeds.get(rd[neighbor_index])
					indices.pop(bisect_left(indices,neighbor_index))
					if len(indices)==0:
						orderseeds.remove(rd[neighbor_index])
				
				#a simple way to deal problem of duplicate value
				indices = orderseeds.get(rd_neighbor)
				if indices is None:
					indices = []
					orderseeds.insert(rd_neighbor,indices)
				indices.insert(bisect_left(indices,neighbor_index),neighbor_index)
				rd[neighbor_index] = rd_neighbor

#implement Procedure ExpandClusterOrder in <<OPTICS: Ordering Points To Identify the Clustering Structure>> Figure 6
def expand_cluster_order(X,index,minpts,rad,cd,rd,order,n_order,processed,kdtree,minpts_neighbor_indices,k_distance,return_of):
	processed[index] = True
	order[n_order] = index
	n_order += 1

	neighbor_indices,neighbor_dists = kdtree.query_radius(X[index,np.newaxis],rad,return_distance=True,sort_results=True)
	neighbor_indices,neighbor_dists = neighbor_indices[0],neighbor_dists[0]
	if return_of:
		minpts_neighbor_indices[index],k_distance[index] = get_minpts_neighbors(X[index,np.newaxis],neighbor_indices,neighbor_dists,minpts,kdtree)

	if neighbor_indices.shape[0]>=minpts:
		cd[index] = neighbor_dists[minpts-1]
		
		orderseeds = RBTree()		#use Red Black Tree to implement priority queue
		orderseed_update(X,orderseeds,cd,rd,processed,index,neighbor_indices,neighbor_dists)
		
		while not orderseeds.is_empty():
			indices = orderseeds.nsmallest(1)[0][1]
			current_object_index = indices.pop()
			if len(indices)==0:
				orderseeds.pop_min()
			
			processed[current_object_index] = True
			order[n_order] = current_object_index
			n_order += 1

			neighbor_indices,neighbor_dists = kdtree.query_radius(X[current_object_index,np.newaxis],rad,return_distance=True,sort_results=True)
			neighbor_indices,neighbor_dists = neighbor_indices[0],neighbor_dists[0]
		
			if return_of:
				minpts_neighbor_indices[current_object_index],k_distance[current_object_index] = get_minpts_neighbors(X[current_object_index,np.newaxis],neighbor_indices,neighbor_dists,minpts,kdtree)

			if neighbor_indices.shape[0]>=minpts:
				cd[current_object_index] = neighbor_dists[minpts-1]
				orderseed_update(X,orderseeds,cd,rd,processed,current_object_index,neighbor_indices,neighbor_dists)

		orderseeds.clear()
	return n_order

#implement Algorithm OPTICS in <<OPTICS: Ordering Points To Identify the Clustering Structure>> Figure 5
def optics(X,minpts,rad,metric=None,return_of=False):
	"""
		OPTICS算法
		参数：
			①X：2D array。存储数据对象
			②minpts、rad：OPTICS算法参数，rad-neighborhood的大小大于等于minpts的数据对象将成为core-point
			③metric：用于计算两个数据对象距离的对象，应使用sklearn.neighbors.dist_metrics中的距离对象。若metric为None，则使用EuclideanDistance
			④return_of：是否返回各个数据对象的outlier factor，默认为False
		
		OPTICS返回cluster-order、core-distance、reachability-distance(以及可选的outlier factor)，通过将这些返回值(除outlier factor外)传入到函数extract_cluster或get_clusters_auto中，可以对数据对象进行聚类，后者进行分层聚类
	"""
	metric = EuclideanDistance() if metric is None else metric
	N = X.shape[0]
	
	order,n_order = np.empty(N,np.uint32),0
	cd,rd = np.empty(N,np.float64),np.empty(N,np.float64)
	cd[:],rd[:] = OPTICS_UNDEFINED,OPTICS_UNDEFINED
	processed = np.zeros(N,np.bool8)
	kdtree = KDTree(X,metric=metric)
	
	minpts_neighbor_indices,k_distance = (np.empty(N,np.object),np.empty(N,np.float64)) if return_of else (None,None)

	_indices_ = np.random.choice(range(N),replace=False,size=N)
	for index in _indices_:
		if not processed[index]:
			n_order = expand_cluster_order(X,index,minpts,rad,cd,rd,order,n_order,processed,kdtree,minpts_neighbor_indices,k_distance,return_of)
	
	if return_of:
		of = get_of(X,minpts,metric,minpts_neighbor_indices,k_distance)

	return (order,cd,rd,of) if return_of else (order,cd,rd)

#implement Algorithm ExtractDBSCAN-Clustering in <<OPTICS: Ordering Points To Identify the Clustering Structure>> Figure 8
def extract_clusters(minpts,rad,order,cd,rd):
	cluster_ids = np.empty(len(order),np.int16)
	current_cluster_id = -1
	for object_index in order:
		if rd[object_index] > rad:
			if less_equal(cd[object_index],rad):
				current_cluster_id += 1
				cluster_ids[object_index] = current_cluster_id
			else:
				cluster_ids[object_index] = -1
		else:
			cluster_ids[object_index] = current_cluster_id
	return cluster_ids

