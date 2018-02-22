import numpy as np
#与numpy.unique相比，增加了sorted关键字
#若sorted=True，则unique假设ar已进行排序
def unique(ar,return_index=False,return_inverse=False,return_counts=False, sorted=False):
	ar = np.asanyarray(ar).flatten()
	optional_indices = return_index or return_inverse
	optional_returns = optional_indices or return_counts

	if ar.size == 0:
		if not optional_returns:
			ret = ar
		else:
			ret = (ar,)
			if return_index:
				ret += (np.empty(0, np.bool),)
			if return_inverse:
				ret += (np.empty(0, np.bool),)
			if return_counts:
				ret += (np.empty(0, np.intp),)
		return ret

	if optional_indices:
		if sorted:
			perm = np.arange(ar.shape[0],dtype=np.uint32)
		else:
			perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
		aux = ar[perm]
	else:
		if not sorted:
			ar.sort()
		aux = ar
	flag = np.concatenate(([True], aux[1:] != aux[:-1]))

	if not optional_returns:
		ret = aux[flag]
	else:
		ret = (aux[flag],)
		if return_index:
			ret += (perm[flag],)
		if return_inverse:
			iflag = np.cumsum(flag) - 1
			inv_idx = np.empty(ar.shape, dtype=np.intp)
			inv_idx[perm] = iflag
			ret += (inv_idx,)
		if return_counts:
			idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
			ret += (np.diff(idx),)
	return ret

from math import log
from scipy.stats._multivariate import _squeeze_output,_PSD,_LOG_2PI

def _logpdf( x, mean, prec_U, log_det_cov, rank, coef=1, diagonal=False, identity=False):
	
	dot = (x-mean) if identity else (x-mean)*prec_U if diagonal else np.dot((x-mean),prec_U)
	square_dot = np.square(dot)
	maha = np.sum(square_dot, axis=-1)/coef
	log_pdf = -0.5 * (rank * (_LOG_2PI + log(coef)) + log_det_cov + maha)
	return log_pdf

def logpdf( x, mean=None, cov=None, allow_singular=False, coef=1, psd=None, return_psd=False):
	if mean is None:
		mean = np.zeros(x.shape[-1],dtype=np.float64)
	if cov is None:
		cov = np.eye(x.shape[-1],dtype=np.float64)
	if psd is None:
		psd = _PSD(cov, allow_singular=allow_singular)
	out = _logpdf(x, mean, psd.U, psd.log_pdet, psd.rank, coef)
	return (_squeeze_output(out),psd) if return_psd else _squeeze_output(out)
