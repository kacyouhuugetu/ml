from .dbscan import dbscan,predict_dbscan
from .clique import clique,predict_clique
from .clarans import clarans,predict_clarans

class cluster:
	def __init__(self,cluster_algorithm,*cluster_algorithm_args,**cluster_algorithm_kwargs):
		"""
			聚类接口，由于不同的聚类算法有不同的算法结构，在这里只提供简单的聚类方法。
			参数：
				①cluster_algorithm：str，表示所使用的聚类算法。当前可使用的聚类算法有'dbscan'、'clarans'、'clique'
				②*cluster_algorithm_args,**cluster_algorithm_kwargs：传入聚类算法的参数
		"""
		if cluster_algorithm == 'dbscan':
			self._train_algorithm,self._predict_algorithm = dbscan,predict_dbscan
		elif cluster_algorithm == 'clarans':
			self._train_algorithm,self._predict_algorithm = clarans,predict_clarans
		elif cluster_algorithm == 'clique':
			self._train_algorithm,self._predict_algorithm = clique,predict_clique
		else:
			raise ValueError("cluster_algorithm currently must be 'dbscan'、'optics'、'clarans' or 'clique'")
		self._cluster_algorithm_args,self._cluster_algorithm_kwargs = cluster_algorithm_args,cluster_algorithm_kwargs

	def train(self,data):
		cluster_ids,self._predict_info = self._train_algorithm(data,*self._cluster_algorithm_args,return_predict_info=True,**self._cluster_algorithm_kwargs)
		return cluster_ids

	def predict(self,data):
		return self._predict_algorithm(data,*self._predict_info)

