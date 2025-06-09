import numpy as np


def top_k(lst, k):
    return sorted(list(enumerate(lst)), key=lambda x: x[1], reverse=True)[:k]

def inf_tie_break(cen_val, centers,k):
    # cen_val is a list of function values
    # centers is a list of centers
    # there are total 2*d faces of the hypercube. x=0, x=1, y=0, y=1, z=0, z=1 and so on.
    # we need to find the distance of the center to each of the faces.

    min_values = np.min(np.hstack((np.abs(centers), np.abs(centers-1))),axis=1)
    # make min_values like a indexed list
    indexed_min_values = list(enumerate(min_values))
    # min_values is a list of minimum distances to the faces. it is of shape (n_points,1)
    mod_cen_val =  np.array([element[0][0] if element != -np.inf else element for element in cen_val ])
    finite_values = mod_cen_val[np.isfinite(mod_cen_val)]

    # having index. since index is important
    indexed_cen_val = list(enumerate(mod_cen_val))
    if len(finite_values) >=k:
        return sorted(indexed_cen_val, key=lambda x: x[1], reverse=True)[:k]
    else:
        # all the finite value will be returned
        index_finite_values = [element for element in indexed_cen_val if element[1] != -np.inf]
        # remaining values will be return based on the min_values
        # remove the finite values indices from the min_values indices
        index_remaining_values = [element for element in indexed_min_values if element[0] not in [element[0] for element in index_finite_values]]
        remaining_values = sorted(index_remaining_values, key=lambda x: x[1], reverse=False)[:k-len(finite_values)]
        return index_finite_values + remaining_values
#        print("There are not enough finite values")

def svd_reduction(directions,d):

	tol = 1e-9
	u, s, vh = np.linalg.svd(directions, full_matrices=True)
	nonzero_indices = np.where(s > tol)[0]
	nonzero_directions = u[:,nonzero_indices]
	if nonzero_directions.shape[1] > d:
		red_directions = nonzero_directions[:,0:d]
	else:
		red_directions = nonzero_directions
	# look at the directions where the singular values are not zero.
	return red_directions