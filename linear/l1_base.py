def get_x_nz_indices(X,variable_index,sparse,is_allclose_zero,slice_all):
	if sparse:
		start_index,end_index = X[1][variable_index],X[1][variable_index+1]
		if start_index==end_index:
			return None,None
		x = X[0][start_index:end_index]
		x_nz_indices = X[2][start_index:end_index]
	else:
		if is_allclose_zero[variable_index]:
			return None,None
		x = X[:,variable_index]
		x_nz_indices = slice_all
	return x,x_nz_indices

