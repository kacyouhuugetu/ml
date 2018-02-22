from random import choice
from bisect import bisect_left
from ..base import isclose,less_equal,linear_kernel,gaussian_kernel
import numpy as np

#根据alpha值和y值确定样本所属的index sets
def get_I_index(alpha,C,y,svm_l1_loss=True):
	if isclose(alpha,0):
		return 1 if y==1 else 4
	elif svm_l1_loss and isclose(alpha,C):
		return 3 if y==1 else 2
	else:
		return 0

#由于维护一个fcache来存储index set为I_0的样本的F值
#故当一个样本离开I_0或一个样本进入I_0，都需要删除或添加fcache的元素
def update_I0_index(index,F,I_index,I_index_old,I_0,fcache):
	if I_index != I_index_old:
		I0_index = bisect_left(I_0,index)
		if I_index_old == 0:
			I_0.pop(I0_index)
			fcache.pop(I0_index)
			I0_index = -1
		if I_index == 0:
			I_0.insert(I0_index,index)
			fcache.insert(I0_index,F)
	else:
		I0_index = bisect_left(I_0,index) if I_index==0 else -1
	return I0_index

#见论文<<Improvements to Platt’s SMO Algorithm for SVM Classifier>> Appendix procedure takeStep(i1,i2)
def take_step(i1,i2,F1,F2,X,y,alpha,C,b_low,i_low,b_up,i_up,individual_C,I_0,fcache,svm_l1_loss,kernel,kernel_args,K,eps):
	if i1==i2:
		return False,b_low,i_low,b_up,i_up

	x1,x2,y1,y2 = X[i1],X[i2],y[i1],y[i2]
	alpha1_old,alpha2_old = alpha[i1],alpha[i2]
	C1,C2 = (C[i1],C[i2]) if individual_C else (C,C)
	ki1,ki2 = ( kernel(X,x1,*kernel_args),kernel(X,x2,*kernel_args) )  if K is None else (K[i1],K[i2])
	if not svm_l1_loss:
		ki1[i1] += (1/(2*C1))
		ki2[i2] += (1/(2*C2))
	k11,k22,k12 = ki1[i1],ki2[i2],ki1[i2]

	if svm_l1_loss:
		if y1!=y2:
			L,H = max(0,alpha2_old-alpha1_old),min(C2,C1+alpha2_old-alpha1_old)
		else:
			L,H = max(0,alpha2_old+alpha1_old-C1),min(C2,alpha2_old+alpha1_old)
	else:
		if y1!=y2:
			L,H = max(0,alpha2_old-alpha1_old),np.inf
		else:
			L,H = 0,alpha2_old+alpha1_old
	if isclose(L,H):
		return False,b_low,i_low,b_up,i_up

	eta = k11*k22-2*k12
	s = y1*y2
	if eta>0:
		alpha2 = alpha2_old + y2*(F1-F2)/eta
		alpha2 = L if alpha2<L else H if alpha2>H else alpha2
	#当eta<=0，任何alpha∈(0,C)都不是最优值
	#因此我们计算端点alpha=0和端点alpha=C时的目标函数值来确定最优alpha值
	#见论文<<Sequential Minimal Optimization:A Fast Algorithm for Training Support Vector Machines>>式(19)
	elif svm_l1_loss:
		f1 = y1*F1 - alpha1_old*k11 - s*alpha2_old*k12
		f2 = y2*F2 - alpha2_old*k22 - s*alpha1_old*k12
		L_ = alpha1_old + s*(alpha2_old-L)
		H_ = alpha1_old + s*(alpha2_old-H)
		obj_L = L_*f1 + L*f2 + 0.5*L_**2*k11 + 0.5*L**2*k22 + s*L*L_*k12
		obj_H = H_*f1 + H*f2 + 0.5*H_**2*k11 + 0.5*H**2*k22 + s*H*H_*k12
		alpha2 = L if obj_L<obj_H-eps else H if obj_H<obj_L-eps else alpha2_old
	else:
		alpha2 = L

	if abs(alpha2-alpha2_old) < eps*(alpha2+alpha2_old+eps):
		return False,b_low,i_low,b_up,i_up
	
	alpha1 = alpha1_old + s*(alpha2_old-alpha2)
	alpha[i1],alpha[i2] = alpha1,alpha2
	
	diff_alpha1,diff_alpha2 = alpha1-alpha1_old,alpha2-alpha2_old
	F1 += y1*diff_alpha1*k11 + y2*diff_alpha2*k12
	F2 += y1*diff_alpha1*k12 + y2*diff_alpha2*k22
	for I0_index in range(len(I_0)):
		sample_index = I_0[I0_index]
		fcache[I0_index] += ( y1*diff_alpha1*ki1[sample_index] + y2*diff_alpha2*ki2[sample_index] )
	
	i1_I_index_old,i2_I_index_old = get_I_index(alpha1_old,C1,y1),get_I_index(alpha2_old,C2,y2)
	i1_I_index,i2_I_index = get_I_index(alpha1,C1,y1),get_I_index(alpha2,C2,y2)
	i1_I0_index = update_I0_index(i1,F1,i1_I_index,i1_I_index_old,I_0,fcache)
	i2_I0_index = update_I0_index(i2,F2,i2_I_index,i2_I_index_old,I_0,fcache)
	
	if len(fcache)>0:
		I0_min,I0_max = min(fcache),max(fcache)
		I0_min_index,I0_max_index = fcache.index(I0_min),fcache.index(I0_max)
	else:
		I0_min,I0_min_index,I0_max,I0_max_index = np.inf,None,-np.inf,None

	min_F,max_F,min_F_index,max_F_index = (F1,F2,i1,i2) if F1<F2 else (F2,F1,i2,i1)
	if I0_min<min_F:
		b_up,i_up = I0_min,I_0[I0_min_index]
	else:
		b_up,i_up = min_F,min_F_index
	if I0_max>max_F:
		b_low,i_low = I0_max,I_0[I0_max_index]
	else:
		b_low,i_low = max_F,max_F_index

	return True,b_low,i_low,b_up,i_up

#见论文<<Improvements to Platt’s SMO Algorithm for SVM Classifier>> Appendix procedure examineExample(i2)
def examine_example(i2,X,y,alpha,C,b_low,i_low,b_up,i_up,individual_C,I_0,fcache,svm_l1_loss,kernel,kernel_args,K,tol,eps):
	y2 = y[i2]
	alpha2 = alpha[i2]
	C2 = C[i2] if individual_C else C
	i2_I_index = get_I_index(alpha2,C2,y2)

	if i2_I_index == 0:
		i2_I0_index = bisect_left(I_0,i2)
		F2 = fcache[i2_I0_index]
	else:
		#计算F值
		#当loss为Hinge-Loss时，F值的计算见论文<<Improvements to Platt’s SMO Algorithm for SVM Classifier>>第2节
		#当loss为Squared Hinge-Loss时，F值的计算见论文<<Comparison of L1 and L2 Support Vector Machines>>式(28)
		ki2 = kernel(X,X[i2],*kernel_args) if K is None else K[i2] 
		if not svm_l1_loss:
			ki2[i2] += 1/(2*C2)
		F2 = np.dot(alpha,y*ki2)-y2
		if i2_I_index<=2:
			if F2<b_up:
				b_up,i_up = F2,i2
		elif i2_I_index<=4:
			if F2>b_low:
				b_low,i_low = F2,i2

	optimality = True
	if i2_I_index<=2:
		if F2 < b_low-2*tol:
			optimality = False
			i1,F1 = i_low,b_low
	if i2_I_index==0 or i2_I_index>=3:
		if F2 > b_up+2*tol:
			optimality = False
			i1,F1 = i_up,b_up
	if optimality:
		return False,b_low,i_low,b_up,i_up

	if i2_I_index==0:
		i1,F1 = (i_low,b_low) if b_low-F2>F2-b_up else (i_up,b_up)
	
	return take_step(i1,i2,F1,F2,X,y,alpha,C,b_low,i_low,b_up,i_up,individual_C,I_0,fcache,svm_l1_loss,kernel,kernel_args,K,eps)

def SMO(X,y,alpha,sample_weights,C,kernel,kernel_args,svm_l1_loss,tol,max_iter,K,buf_ratio):
	"""
		Sequential Minimal Optimization算法。具体见论文<<Sequential Minimal Optimization:A Fast Algorithm for Training Support Vector Machines>>、<<Improvements to Platt’s SMO Algorithm for SVM Classifier>>和<<Comparison of L1 and L2 Support Vector Machines>>
	"""
	N,p = X.shape
	indices = np.arange(N,dtype=np.uint32)

	if alpha is None:
		I_0,fcache = [],[]
		b_up,i_up = -1,choice(indices[y==1])
		b_low,i_low = 1,choice(indices[y!=1])
		alpha = np.zeros(N,np.float64)
	
	#计算I_0、i_low、b_low、i_up、b_up等
	#具体见论文<<Improvements to Platt’s SMO Algorithm for SVM Classifier>>式(5)
	else:
		alpha_mul_y = alpha*y
		if not K is None:
			F = np.dot(K,alpha_mul_y) - y

		else:
			buf_size = min(N,max(1,int(N*buf_ratio)))
			F = np.empty(N,np.float64)

			start_index,end_index = 0,buf_size
			while start_index<N:
				K_ = kernel(X[start_index:end_index],X,*kernel_args)
				F[start_index:end_index] = np.dot(K_,alpha_mul_y) - y[start_index:end_index]
				start_index,end_index = end_index,end_index+buf_size
		del alpha_mul_y

		alpha_0_indices,alpha_C_indices = np.isclose(alpha,0.0),np.isclose(alpha,C)
		y_1_indices,y_neg_1_indices = y==1,y==-1
		
		I_0 = indices[np.logical_not(np.logical_or(alpha_0_indices,alpha_C_indices))]
		fcache = F[I_0]
		
		I_1,I_2 = indices[np.logical_and(y_1_indices,alpha_0_indices)],indices[np.logical_and(y_neg_1_indices,alpha_C_indices)]
		I_0_1_2 = np.hstack((I_0,I_1,I_2))
		i_up = I_0_1_2[np.argmin(F[I_0_1_2])]
		b_up = F[i_up]
		del I_1,I_2,I_0_1_2

		I_3,I_4 = indices[np.logical_and(y_1_indices,alpha_C_indices)],indices[np.logical_and(y_neg_1_indices,alpha_0_indices)]
		I_0_3_4 = np.hstack((I_0,I_3,I_4))
		i_low = I_0_3_4[np.argmin(F[I_0_3_4])]
		b_low = F[i_low]
		del I_3,I_4,I_0_3_4
		del F,indices,alpha_0_indices,alpha_C_indices,y_1_indices,y_neg_1_indices
		
		I_0,fcache = list(I_0),list(fcache)

	eps = np.finfo(np.float64).eps

	if not sample_weights is None:
		C = sample_weights*C
		individual_C = True
	else:
		individual_C = False
	
	numchanged,examineAll = False,True
	iter_index = 0
	while (numchanged>0 or examineAll) and iter_index<max_iter:
		iter_index += 1
		numchanged = 0
		sample_indices = range(N) if examineAll else I_0
		for sample_index in sample_indices:
			changed,b_low,i_low,b_up,i_up = examine_example(sample_index,X,y,alpha,C,b_low,i_low,b_up,i_up,individual_C,I_0,fcache,svm_l1_loss,kernel,kernel_args,K,tol,eps)
			numchanged+=changed
			if not examineAll and less_equal(b_low,b_up+2*tol):
				numchanged = 0
				break
		if examineAll:
			examineAll = False
		elif numchanged==0:
			examineAll = True
		
	return alpha,b_low,b_up

