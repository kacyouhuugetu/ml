from ..base import logistic_fun,log_logistic_fun
import numpy as np

_LOSS_TYPE_SVM_L1_,_LOSS_TYPE_SVM_L2_,_LOSS_TYPE_LOGISTIC_ = 0,1,2

_loss_svm_l1 = lambda y,z,w=None:np.sum(1-y*z) if w is None else np.dot(w,1-y*z)
_loss_svm_l2 = lambda y,z,w=None:np.sum((z-y)**2) if w is None else np.dot(w,(z-y)**2)
_loss_logistic = lambda y,z,w=None:-np.sum(log_logistic_fun(y*z)) if w is None else -np.dot(w,log_logistic_fun(y*z))

#w为sample weights
_loss_svm_l1_gradient = lambda X_T,y,z,w=None:-X_T.dot(y) if w is None else -X_T.dot(w*y)
_loss_svm_l2_gradient = lambda X_T,y,z,w=None:X_T.dot(z-y) if w is None else X_T.dot(w*(z-y))
_loss_logistic_gradient = lambda X_T,y,z,w=None:-X_T.dot(logistic_fun(-y*z)*y) if w is None else -X_T.dot(w*logistic_fun(-y*z)*y)


def _get_nl(w,X,y,sample_weights,svm_loss):
	z = X.dot(w)
	if svm_loss:
		nonzero_loss_indices = y*z<1
		nl_X,nl_y,nl_z = X[nonzero_loss_indices],y[nonzero_loss_indices],z[nonzero_loss_indices]
		nl_sample_weights = None if sample_weights is None else sample_weights[nonzero_loss_indices]
	else:
		nonzero_loss_indices = None
		nl_X,nl_y,nl_z,nl_sample_weights = X,y,z,sample_weights
	return nl_X,nl_y,nl_z,nl_sample_weights,nonzero_loss_indices

