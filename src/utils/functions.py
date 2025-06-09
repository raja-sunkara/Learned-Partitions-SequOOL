import numpy as np
#from shapely.geometry import Point, Polygon
#import math

# from numpy import abs, arange, arctan2, asarray, cos, exp, floor, log, log10, mean
# from numpy import pi, prod, roll, seterr, sign, sin, sqrt, sum, zeros, zeros_like, tan
# from numpy import dot, inner


rng = np.random.default_rng()
np.random.seed(seed=10)

harmonic_number = lambda n: sum(1/i for i in range(1, n+1))

# def lzip(*args):
#     """
#     Zip, but returns zipped result as a list.
#     """
#     return list(zip(*args))

def print_status(d,n,h,i_max,x,value,trees):

    x_str = ""
    for i in range(d):
        x_str += f" {x[i]}"
    print(f"{n}: sampling tree {trees}, ({h},{i_max}): f({x_str.lstrip()}) = {value}\n")

# definition of the function to be optimized


# # Define a function to check if two lines intersect in a unit square
# def check_intersection(line1, line2):
#     # Find the intersection point of the two lines
#     det = line1[0]*line2[1] - line1[1]*line2[0]
#     if det == 0:
#         # The lines are parallel, so they don't intersect
#         return False
#     else:
#         x = (line2[1]*line1[2] - line1[1]*line2[2]) / det
#         y = (line1[0]*line2[2] - line2[0]*line1[2]) / det
#         # Check if the intersection point is inside the unit square
#         if 0 <= x <= 1 and 0 <= y <= 1:
#             return True
#         else:
#             return False

# # def two_lines_intersect():
# #     while True:
# #         line1 = generate_random_line()
# #         line2 = generate_random_line()
# #         if check_intersection(line1, line2):
# #             break
# #     return line1, line2


# def sin1():
#     sin1 = lambda value: (np.sin(13 * value) * np.sin(27 * value) / 2.0 + 0.5) # used in ICML 2013 paper
#     fmax = 0.975599143811574975870826165191829204559326171875
#     return sin1, fmax

# def garland():
#     garland = lambda x: 4*x*(1-x)*(0.75+0.25*(1-np.sqrt(abs(np.sin(60*x)))))  # used in ICML 2013 paper
#     fmax = 0.997772313413222  # this value is from the matlab soo code. will need to check if this is correct.

#     return garland, fmax

# def warped_sine():

#     warped_sine = lambda x: 0.5 * np.sin(np.pi * np.log2(2 * np.abs(x - 0.5)) + 1) * ((2 * np.abs(x - 0.5)) ** (-np.log10(0.8)) - (2 * np.abs(x - 0.5)) ** (-np.log10(0.3))) - (2 * np.abs(x - 0.5)) ** (-np.log10(0.8))
#     fmax = 0

#     return warped_sine, fmax


# def difficult():
#     difficult = lambda x: 1-np.sqrt(x) + (-x*x +np.sqrt(x) )*(np.sin(1/(x*x*x))+1)/2
#     fmax = None

#     return difficult, fmax

# def three_dim():
#     trial = lambda value: (value[1]-0.5) * (value[2]-0.5)*(value[0]-0.5)
#     fmax = 1.0/8
#     return trial, fmax

# def two_dim():
#     trial = lambda value: (value[1]-0.5) *(value[0]-0.5)
#     fmax = 1.0/4
#     return trial, fmax

# def four_dim():
#     trial = lambda value: (value[1]-0.5) *(value[0]-0.5) * (value[2]-0.5) * (value[3]-0.5)
#     fmax = 1.0/16
#     return trial, fmax

# class Triangle_wave():
#     def __init__(self,num_hidden,n_dim) -> None:
#         super(Triangle_wave, self).__init__()
#         self.num_hidden = num_hidden
#         self.fmax = num_hidden
#         self.dim = n_dim
        
#         if n_dim ==2 :
#             if num_hidden == 2:
#                 #self.W1 = np.array([[1,1],[1,0.5],[-0.6,-0.4]])  # cfg1
#                 #self.W1 = np.array([[1,1],[1,-1],[-0.6,-0.4]])
#                 #self.W1 = np.array([[4,1],[1,4],[-1.25,-1.25]])  # cfg2
#                 #self.W1 = np.array([[3,1],[1,3],[-1.33,-1.33]])  # cfg3
#                 self.W1 = np.array([[5,1],[1,5],[-1.8,-1.8]])  # cfg4

#                 #self.W1 = np.array([[1,2],[1,2],[-0.6,-1.2]]) # linear depedent ridge directions

#                 #self.W1 = rng.random((self.dim + 1, self.num_hidden)) * 2 - 0.5
#                 #self.W1 = np.array([[1,2],[1,2],[-0.6,-0.4]])  # cfg1


#                 # l1,l2 = two_lines_intersect()
#                 # self.W1 = np.vstack((l1,l2)).T

#                 # self.W1 = np.random.randn(self.dim, self.num_hidden) * 0.5
#                 # # find intercept so that all pass through (0.3)*dim
#                 # intercept = -np.dot(self.W1.T, np.array([0.3]*self.dim))
#                 # self.W1 = np.vstack((self.W1, intercept))

#                 #cfg5

#                 #self.W1 = np.array([[1,1.1],[1,1],[-0.6,-0.62]])  # cfg6 (0.2,0.4)               
#                 #self.W1 = np.array([[1,2],[1,4],[-1,-1]]) # cfg7 (0.2,0.4)

                
#                 # self.W1 = np.array([[3,1],[1,3]])  # cfg 7
#                 # # # find intercept so that all pass through (0.42127)*dim
#                 # intercept = -np.dot(self.W1.T, np.array([0.42127]*self.dim))
#                 # self.W1 = np.vstack((self.W1, intercept))

#                 # self.W1 = np.array([[1,1],[1,0.5]])  # cfg8
#                 # intercept = -np.dot(self.W1.T, np.array([0.42127]*self.dim))
#                 # self.W1 = np.vstack((self.W1, intercept))
#             elif num_hidden == 1:
#                 #self.W1 = np.array([[1],[1],[-0.6]]) # cfg1

#                 self.W1 = np.array([[1.0/0.8],[1.0/0.9],[-2]])  # cfg2

#             elif num_hidden == 3:
#                 # self.W1 = np.array([[1,1,3],[1,0.5,1],[-0.6,-0.4,-0.3]])  

#                 # one passin through (0.25,0.25)
#                 self.W1 = np.array([[4,4,1],[4,-4,0],[-2,0,-0.25]])

#                 # one apssigng through (0.5,0.8)
#                 # self.W1 = np.array([[2,2,0],[1.25,0,1.25],[-2,-1,-1]])
#                 # self.W1 = np.array([[2,-2,0],[1.25,1.25,1.25],[-2,0,-1]])    

                

#                 #cfg3            

#             elif num_hidden == 4:
#                 #self.W1 = np.array([[1,1,3],[1,0.5,1],[-0.6,-0.4,-0.3]])  
#                 #one passin through (0.25,0.25)
#                 self.W1 = np.array([[4,4,1,0],[4,-4,0,1],[-2,0,-0.25,-0.25]])

#             elif num_hidden == 5:
#                 self.W1 = np.array([[4,4,1,0,8],[4,-4,0,1,4],[-2,0,-0.25,-0.25,-3]])


#             elif num_hidden >5:
#                 self.W1 = rng.random((self.dim + 1, self.num_hidden)) * 2 - 0.5

#         elif n_dim == 3:
#             if num_hidden == 3:
#                 self.W1 = np.array([[5,-5,5],[5,5,0],[5,5,0],[-3,-1,-1]]) #(0.2,0.2,0.2) cfg2

#                 #self.W1 = np.array([[1,1,1],[1,-1,1],[1,1,-1],[-0.75,-0.25,-0.25]]) # (0.25,0.25,0.25)


#                 #self.W1 = np.random.randn(self.dim, self.num_hidden) * 0.5
#                 # find intercept so that all pass through (0.3)*dim
#                 #intercept = -np.dot(self.W1.T, np.array([0.3]*self.dim))
#                 #self.W1 = np.vstack((self.W1, intercept))

#                 #cfg3

#         elif n_dim == 4:
#             if num_hidden == 4:
#                 self.W1 = np.array([[1,3,4,1],[1,2,1,-1],[1,3,4,1],[1,2,1,-1],[-0.8,-2,-2,0]])
#                 #self.W1 = np.array([[1,1,-1,1],[1,-1,-1,-1],[1,1,-1,-1],[1,-1,1,-1],[-0.8,0,0.4,0]]) # cfg2

#                 #self.W1 = np.random.randn(self.dim, self.num_hidden) * 0.5
#                 # find intercept so that all pass through (0.3)*dim
#                 #intercept = -np.dot(self.W1.T, np.array([0.3]*self.dim))
#                 #self.W1 = np.vstack((self.W1, intercept))

#                 #cfg3

                

#             if num_hidden == 2:
#                 self.W1 = np.array([[1,3],[1,2],[1,3],[1,2],[-0.8,-2]])

#         elif n_dim == 6 and num_hidden == 6:
#             #self.W1 = np.array([[1,1,1,1,1,1],[1,-1,-1,-1,1,1],[1,1,1,1,-1,1],[1,-1,1,-1,-1,-1],[1,1,1,1,1,-1],[1,-1,1,1,1,-1],[-1.5,0,-1,-0.5,-0.5,0]]) #(0.25)*6 cfg1

#             self.W1 = np.random.randn(self.dim, self.num_hidden) * 0.5
#             # find intercept so that all pass through (0.3)*dim
#             intercept = -np.dot(self.W1.T, np.array([0.3]*self.dim))
#             self.W1 = np.vstack((self.W1, intercept))

#             #cfg2

#         elif n_dim == 8:
#             if num_hidden == 8:
#                 self.W1 = np.array([[1,1,1,1,1,1,1,-1],[1,-1,-1,-1,1,1,1,-1],[1,1,1,1,-1,1,1,-1],[1,-1,1,-1,-1,-1,1,-1],[1,1,1,1,1,-1,1,-1],[1,-1,1,1,1,-1,-1,-1],[1,1,1,1,1,1,-1,-1],[1,-1,1,1,1,-1,-1,1],[-2,0,-1.5,-1,-1,0,-0.5,1.5]]) #(0.25)*8

#             elif num_hidden == 4:
#                 self.W1 = np.array([[1,1,1,1],[1,-1,-1,-1],[1,1,1,1],[1,-1,1,-1],[1,1,1,1],[1,-1,1,1],[1,1,1,1],[1,-1,1,1],[-2,0,-1.5,-1]]) #(0.25)*8

#         elif n_dim == 7:
#             if num_hidden == 7:
#                 self.W1 = np.random.randn(self.dim, self.num_hidden) * 0.5
#                 # find intercept so that all pass through (0.3)*dim
#                 intercept = -np.dot(self.W1.T, np.array([0.3]*self.dim))
#                 self.W1 = np.vstack((self.W1, intercept))

#         elif n_dim == 10:
#             if num_hidden == 10:
#                 self.W1 = np.array([[1,1,1,1,1,1,1,1,1,1],[1,-1,-1,-1,1,1,1,1,1,1],[1,1,1,1,-1,1,1,1,1,1],[1,-1,1,-1,-1,-1,1,1,1,1],[1,1,1,1,1,-1,1,1,1,1],[1,-1,1,1,1,-1,-1,-1,1,1],[1,1,1,1,1,1,-1,-1,1,1],[1,-1,1,1,1,-1,-1,-1,1,1],[1,1,1,1,1,1,1,1,-1,-1],[1,-1,1,1,1,-1,-1,-1,-1,-1],[-1.5,0,-1,-0.5,-0.5,0,0,0,0,0]]) #(0.25)*10

#             elif num_hidden == 5:
#                 self.W1 = np.array([[1,1,1,1,1],[1,-1,-1,-1,1],[1,1,1,1,-1],[1,-1,1,-1,-1],[1,1,1,1,1],[1,-1,1,1,1],[1,1,1,1,1],[1,-1,1,1,1],[1,1,1,1,1],[1,-1,1,1,1],[-1.5,0,-1,-0.5,-0.5]])
            

#         else:
#             self.W1 = rng.random((self.dim + 1, self.num_hidden)) * 2 - 0.5

#             # self.W1 = np.array([[ 0.66677485],[ 0.26026506],[ 1.44410066], [-0.46192814],[ 0.64093076],
#             # [ 0.08451057], [ 0.49163342], [ 0.59341282], [ 0.8169405 ], [-0.42100286], [-0.21434189]])
# #            self.W1 = np.ones((self.dim + 1, self.num_hidden))
            
#             self.W1[:,0] = 4*np.ones(self.dim+1) 
#             self.W1[:,0][-1] = -10
#             #self.W1[:,1] = 4*np.array([1*(-1)**i for i in range(self.dim+1)]) 
#             self.W1[:,1] = 4*np.ones(self.dim+1) 
#             self.W1[:,1][-1] = -10
#             # self.W1[-1] = np.array([-0.7,-0.3])

#         self.W2 = np.ones((self.num_hidden,1))
#         # W2 = rng.random((self.num_hidden + 1, 1)) * 2 - 0.5
#         # W1[:-1, :] /= np.linalg.norm(W1[:-1, :], axis=1, keepdims=True)
#     def __call__(self,X):

#         '''
#         X is num_data by dim_data, 
#         W1 is dim_data + 1 by num_hidden, 
#         W2 is num_hidden + 1 by 1
#         '''

#         num_data = len(X)
#         if X.ndim == 1: 
#             X = X[:, np.newaxis]
#         dim_data = self.W1.shape[0] - 1
#         if X.shape[1] == dim_data: 
#             X = np.hstack((X, np.ones((num_data, 1))))
        
#         hidden_preact = X @ self.W1
#         hidden_act = np.maximum(1-np.abs(hidden_preact), 0) # relu activation
#         # hidden_act = np.hstack((hidden_act, np.ones((num_data, 1))))
#         return (hidden_act @ self.W2)

#     def grid_function_value(self,x):
#         '''
#         x is a list of meshgrid for each dimension
#         '''
#         # Q: Meshgrid for multiple dimensions?
#         # A: https://stackoverflow.com/questions/36013063/construct-mesh-grid-in-3d-python

#         reshaped_x = [x.reshape(-1,1) for x in x]
#         X = np.hstack(reshaped_x)

#         # X = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
#         num_data = len(X)
#         dim_data = self.W1.shape[0] - 1
#         X = np.hstack((X, np.ones((num_data, 1))))
#         hidden_preact = X @ self.W1
#         hidden_act = np.maximum(1-np.abs(hidden_preact), 0) # relu activation
#         return (hidden_act @ self.W2)


# class Triangle_wave2():
#     def __init__(self,num_hidden,n_dim) -> None:
#         super(Triangle_wave2, self).__init__()
#         self.num_hidden = num_hidden
#         self.fmax = num_hidden
#         self.dim = n_dim
#         if n_dim ==2 :
#             if num_hidden == 2:
#                 #self.W1 = np.array([[1.2,10],[1,1],[-0.6,-0.8]])
#                 #self.W1 = np.array([[5,1],[1,5],[-1.8,-1.8]])  # cfg4

#                 #self.W1 = np.array([[1,1],[1,0.5],[-0.6,-0.4]])  # cfg1 (0.2,0.4)
#                 #self.W1 = np.array([[1,1],[1,-1],[-0.6,-0.4]])
#                 #self.W1 = np.array([[4,1],[1,4],[-1.25,-1.25]])  # cfg2
#                 self.W1 = np.array([[3,1],[1,3],[-1.33,-1.33]])  # cfg3
#                 #self.W1 = np.array([[5,1],[1,5],[-1.8,-1.8]])  # cfg4
                
#             elif num_hidden == 1:
#                 self.W1 = np.array([[1.2],[1],[-0.7]])
#             # W1 = rng.random((self.dim + 1, self.num_hidden)) * 2 - 0.5

#         else:
#             self.W1 = rng.random((self.dim + 1, self.num_hidden)) * 2 - 0.5

#             # self.W1 = np.array([[ 0.66677485],[ 0.26026506],[ 1.44410066], [-0.46192814],[ 0.64093076],
#             # [ 0.08451057], [ 0.49163342], [ 0.59341282], [ 0.8169405 ], [-0.42100286], [-0.21434189]])
#             self.W1 = np.ones((self.dim + 1, self.num_hidden))
#             self.W1[-1] = -0.7

#         self.W2 = np.ones((self.num_hidden,1))
#         # W2 = rng.random((self.num_hidden + 1, 1)) * 2 - 0.5
#         # W1[:-1, :] /= np.linalg.norm(W1[:-1, :], axis=1, keepdims=True)
#     def __call__(self,x,y):

#         '''
#         x and y are meshgrid arrays of shape (num_points, num_points)
#         '''
#         X = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
#         num_data = len(X)
#         dim_data = self.W1.shape[0] - 1
#         X = np.hstack((X, np.ones((num_data, 1))))
#         hidden_preact = X @ self.W1
#         hidden_act = np.maximum(1-np.abs(hidden_preact), 0) # relu activation
#         return (hidden_act @ self.W2).reshape(x.shape)


# def triangle_wave3(num_hidden,n_dim):
#     ## this function here compute the maximum over the domain.

#     f = Triangle_wave(num_hidden,n_dim)
#     sample_points = 30
#     x = np.meshgrid(*[np.linspace(0, 1, sample_points) for i in range(n_dim)])
#     # x_values = np.linspace(0, 1, sample_points)
#     # y_values = np.linspace(0, 1, sample_points)
    
#     # x_grid, y_grid = np.meshgrid(x_values, y_values)
#     outputs = f.grid_function_value(x)
#     max_value = np.max(outputs)
#     return f, max_value

# def triangle_wave(num_hidden,n_dim):

#     return Triangle_wave(num_hidden,n_dim), num_hidden

# def triangle_wave2(num_hidden,n_dim):

#     return Triangle_wave2(num_hidden,n_dim), num_hidden

# def line_points(x_min, x_max):
#     # Calculate the 1/3 and 2/3 points along the line
#     x1,y1,x2,y2 = x_min[0], x_min[1], x_max[0],x_max[1]
#     p1 = (x1 + 1/3*(x2-x1), y1 + 1/3*(y2-y1))
#     p2 = (x1 + 2/3*(x2-x1), y1 + 2/3*(y2-y1))

#     left_mid = ((p1[0]+x1)/2, (p1[1] + y1)/2)
#     right_mid = ((p2[0]+x2)/2, (p2[1] + y2)/2)
    
#     # Calculate the midpoint of the line
#     midpoint = ((x1+x2)/2, (y1+y2)/2)
    
#     return np.array(p1), np.array(p2), np.array(midpoint), np.array(left_mid), np.array(right_mid)



# def line_intersection_with_unit_square(l):
#     a,b,c = l[0],l[1],l[2]
#     # Find intersection points with x = 0, x = 1, and y = 1
#     x1 = 0
#     y1 = -c / b
#     x2 = 1
#     y2 = (-a - c) / b
#     y3 = 0
#     x3 = -c / a
#     x4 = (-b - c) / a
#     y4 = 1

#     # Filter out points not in positive quadrant or on boundary
#     points = [(x, y) for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] if x >= 0 and y >= 0 and x<=1 and y <=1]

#     return points


# def project_rectangle(rect_coords, v):
#     # Convert rectangle coordinates to numpy array
#     rect_coords = np.array(rect_coords)
    
#     # Compute dot products with vector v
#     dot_products = np.dot(rect_coords, v)
    
#     # Sort dot products in ascending order
#     sorted_indices = np.argsort(dot_products)
    
#     # Extract smallest and largest dot products
#     smallest_dp = dot_products[sorted_indices[:2]]
#     largest_dp = dot_products[sorted_indices[-2:]]
    
#     # Compute projected coordinates
#     smallest_coords = rect_coords[sorted_indices[:2]]
#     largest_coords = rect_coords[sorted_indices[-2:]]
#     projected_coords = np.concatenate((smallest_coords, largest_coords))
    
#     # Compute width and height of projected rectangle
#     width = np.linalg.norm(largest_coords[0] - smallest_coords[0])
#     height = np.linalg.norm(largest_coords[1] - smallest_coords[1])
    
#     return projected_coords, width, height


def rotation_to_original(R1,p1):
    point = p1.copy()
    return np.dot(R1, point.T).reshape(1,-1) + np.array([0.5]*len(R1))

# def point_in_unit_square(point):
#     # Check if point lies within the unit square
#     if 0 <= point[0] <= 1 and 0 <= point[1] <= 1:
#         return True
#     # Check if point lies on one of the unit square edges
#     elif point[0] == 0 or point[0] == 1:
#         return 0 <= point[1] <= 1
#     elif point[1] == 0 or point[1] == 1:
#         return 0 <= point[0] <= 1
#     # If point is outside the unit square, return False
#     else:
#         return False

# def top_k_indices(lst, k):
#     # Create a list of tuples with the element and its index
#     indexed_lst = list(enumerate(lst))
    
#     # Sort the list of tuples by the element in descending order
#     sorted_lst = sorted(indexed_lst, key=lambda x: x[1], reverse=True)
    
#     # Initialize a dictionary to keep track of the indices
#     indices_dict = {}
    
#     # Iterate over the sorted list of tuples, adding indices to the dictionary
#     for i, (val, _) in enumerate(sorted_lst):
#         if val in indices_dict:
#             indices_dict[val].append(i)
#         else:
#             indices_dict[val] = [i]
    
#     # Initialize a list to hold the indices of the top k values
#     top_k_indices = []
    
#     # Iterate over the sorted list of tuples, adding indices to the list
#     for i, (val, _) in enumerate(sorted_lst):
#         if i >= k:
#             break
#         top_k_indices.extend(indices_dict[val][:k-i])
    
#     # Sort the list of indices in ascending order
#     top_k_indices.sort()
    
#     # Return the list of indices
#     return top_k_indices




# def top_k_two_arrays(t1,t2,k):
#     # Concatenate the arrays and sort by 'cen_val'
#     t = np.concatenate((t1, t2), axis=0)
#     t_sorted = t[np.argsort(t)]

#     # Get the top k elements

#     top_k = t_sorted[-k:]

#     # Get the corresponding array numbers and indices
#     result = []
#     for elem in top_k:
#         if np.isin(elem, t1):
#             idx = np.where(t1 == elem)[0][0]
#             result.append(('t1', idx))
#         else:
#             idx = np.where(t2 == elem)[0][0]
#             result.append(('t2', idx))


#     return result


# def sup_val(f,x_min,x_max,h):

#     # Define the number of samples per dimension

#     if h<8:

#         n_samples_x = 100
#         n_samples_y = 100

#     else:
#         n_samples_x = 50
#         n_samples_y = 50

#     # Create a meshgrid of points within the rectangular grid
#     x_values = np.linspace(x_min[0], x_max[0], n_samples_x)
#     y_values = np.linspace(x_min[1], x_max[1], n_samples_y)
#     x_grid, y_grid = np.meshgrid(x_values, y_values)


#     outputs = f([x_grid, y_grid])


#     # Find the maximum value of the evaluated function
#     max_value = np.max(outputs)

#     return max_value

# def sup_val2(f,x_min,x_max,h):

#     # Define the number of sample points
#     n_samples = 10

#     # Get the x and y coordinates of the two points
#     x1, y1 = x_min[0], x_min[1]
#     x2, y2 = x_max[0], x_max[1]

#     # Compute the linearly spaced array of coordinates
#     x_values = np.linspace(x1, x2, n_samples)
#     y_values = np.linspace(y1, y2, n_samples)

#     # Evaluate the function at each point in the array
#     points = np.vstack((x_values, y_values)).T
#     function_values = f(points)

#     # Find the maximum value of the evaluated function
#     max_value = np.max(function_values)

#     return max_value

# def sup_val_parallelogram(f, point1,point2,point3,point4,h):

#     def Random_Points_in_Polygon(polygon, number):
#         points = []
#         minx, miny, maxx, maxy = polygon.bounds
#         while len(points) < number:
#             pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
#             if polygon.contains(pnt):
#                 points.append(pnt)
#         return points

#     def Uniform_Points_in_Polygon(polygon, number):
#         points = []
#         minx, miny, maxx, maxy = polygon.bounds
#         n = int(np.sqrt(number))
#         x_vals = np.linspace(minx, maxx, n)
#         y_vals = np.linspace(miny, maxy, n)
#         for x in x_vals:
#             for y in y_vals:
#                 pnt = Point(x, y)
#                 if polygon.contains(pnt):
#                     points.append(pnt)
#         return points[:number]

#     parallelogram = Polygon([point1, point2, point3, point4])
#     points = Uniform_Points_in_Polygon(parallelogram, 100)

#     listarray = []
#     for pp in points:
#         listarray.append([pp.x, pp.y])
#     nparray = np.array(listarray)

#     suprem = np.max(f(nparray))
#     return suprem

# def m_val(f,x):

#     return abs(f.W1[0]*x[0] + f.W1[1]*x[1]+f.W1[2])


# def sup_val3(f,R1,x_min,x_max,h):

#     # Define the number of samples per dimension

#     if h<8:

#         n_samples_x = 10
#         n_samples_y = 10

#     else:
#         n_samples_x = 4
#         n_samples_y = 4

#     # Create a meshgrid of points within the rectangular grid
#     x_values = np.linspace(x_min[0], x_max[0], n_samples_x)
#     y_values = np.linspace(x_min[1], x_max[1], n_samples_y)
#     x_grid, y_grid = np.meshgrid(x_values, y_values)

#     grid_points = np.stack([x_grid, y_grid], axis=-1)

#     # Apply the transformation to the grid points
#     transformed_grid = np.matmul(grid_points, R1.T)+ np.array([0.5, 0.5])

#     # Evaluate the function at the transformed grid points
#     outputs = f(transformed_grid[:, :, 0], transformed_grid[:, :, 1])


#     # outputs = f(x_grid, y_grid)

#     # Find the maximum value of the evaluated function
#     max_value = np.max(outputs)

#     return max_value


# def maximum_line_value(f, min_point, max_point):
#     """
#     Evaluate a linear function on the four corners of a rectangle and return the maximum value.

#     Args:
#     f (object): A linear function with parameters f.W1[0], f.W1[1], and f.W1[2].
#     min_point (tuple): The minimum x and y coordinates of the rectangle as a tuple (min_x, min_y).
#     max_point (tuple): The maximum x and y coordinates of the rectangle as a tuple (max_x, max_y).

#     Returns:
#     float: The maximum value of the linear function evaluated on the four corners of the rectangle.
#     """
#     corners = [(min_point[0], min_point[1]), (min_point[0], max_point[1]), (max_point[0], min_point[1]), (max_point[0], max_point[1])]
#     values = [abs(f.W1[0]*corner[0] + f.W1[1]*corner[1] + f.W1[2]) for corner in corners]

#     return max(values)

# def maximum_f_value_square(f, min_point, max_point,n):
#     minx, miny, maxx, maxy = min_point[0], min_point[1], max_point[0], max_point[1]

#     x_vals = np.linspace(minx, maxx, n)
#     y_vals = np.linspace(miny, maxy, n)
#     xx, yy = np.meshgrid(x_vals, y_vals)
    
#     outputs = f(xx, yy)

#     sub_optimality = np.max(2 - outputs)

#     return sub_optimality


# def maximum_f_value(f, point1,point2,point3,point4,fmax=1,evaluate_corners=False):


#     def Uniform_Points_in_Polygon(polygon, number):
#         points = []
#         unit_square = Polygon([[0,0], [1,0], [1,1], [0,1]])
#         minx, miny, maxx, maxy = polygon.bounds
#         n = int(np.sqrt(number))
#         x_vals = np.linspace(minx, maxx, n)
#         y_vals = np.linspace(miny, maxy, n)
#         for x in x_vals:
#             for y in y_vals:
#                 pnt = Point(x, y)
#                 if polygon.contains(pnt) or polygon.touches(pnt):
#                     points.append(pnt)
#         return points[:number]
    
#     if evaluate_corners:
#         temp = np.max(fmax - f(np.array([point1,point2, point3, point4]).T))
#         return temp
#     else:
    
#         parallelogram = Polygon([point1, point2, point3, point4])
#         points = Uniform_Points_in_Polygon(parallelogram, 10000)

#         listarray = []
#         for pp in points:
#             listarray.append([pp.x, pp.y])
#         nparray = np.array(listarray).T

#         sub_optimality = np.max(fmax - f(nparray))

#         return sub_optimality

# def maximum_line_value_two_points(f, min_point, max_point):
#     """
#     Evaluate a linear function on the four corners of a rectangle and return the maximum value.

#     Args:
#     f (object): A linear function with parameters f.W1[0], f.W1[1], and f.W1[2].
#     min_point (tuple): The minimum x and y coordinates of the rectangle as a tuple (min_x, min_y).
#     max_point (tuple): The maximum x and y coordinates of the rectangle as a tuple (max_x, max_y).

#     Returns:
#     float: The maximum value of the linear function evaluated on the four corners of the rectangle.
#     """
#     corners = [(min_point[0], min_point[1]), (max_point[0], max_point[1])]
#     values = [abs(f.W1[0]*corner[0] + f.W1[1]*corner[1] + f.W1[2]) for corner in corners]
#     return max(values)

# def camel6():

#     # for came function. the bounds are [-3,3] x [-2,2]
#     #given x1, x2 by the sequool algoithm, we need to transform them to [-3,3] x [-2,2]
#     # x1_prime = -3 + 6*x1
#     # x2_prime = -2 + 4*x2


#     camel6 = lambda xx: -(4-2.1*xx[0]**2+(xx[0]**4)/3) * xx[0]**2 - xx[0]*xx[1] - (-4+4*xx[1]**2) * xx[1]**2

#     camel6_transformed = lambda xx: camel6(np.array([6*xx[0]-3, 4*xx[1]-2]))

#     transform = lambda xx: np.array([6*xx[0]-3, 4*xx[1]-2])
#     x_star = np.array([0.0898, -0.7126])
#     fmax = camel6(x_star)
    

#     return camel6, transform, fmax, x_star

# def bukin6():

#     # for bukin function. the bounds are [-15, -5] x [-3, 3]
#     #given x1, x2 by the sequool algoithm, we need to transform them to [-15, -5] x [-3, 3]
#     # x1_prime = -15 + 10*x1
#     # x2_prime = -3 + 6*x2

#     transform = lambda xx: np.array([10*xx[0]-15, 6*xx[1]-3])
#     fmax = 0
#     x_star = np.array([-10, 1])

#     bukin6 = lambda xx: -  ( 100 * np.sqrt(abs(xx[1] - 0.01*xx[0]**2)) + 0.01 * abs(xx[0]+10))

#     return bukin6, transform, fmax, x_star

# def colville():

#     colville = lambda xx: \
#        - ( 100 * (xx[0]**2 - xx[1])**2 + \
#         (xx[0] - 1)**2 + \
#         (xx[2] - 1)**2 + \
#         90 * (xx[2]**2 - xx[3])**2 + \
#         10.1 * ((xx[1] - 1)**2 + (xx[3] - 1)**2) - \
#         19.8 * (xx[1] - 1) * (xx[3] - 1) )

#     # this function is defined on [-10,10]
#     # x1_prime = -10 + 20*x1
#     # x2_prime = -10 + 20*x2
#     transform = lambda xx: np.array([20**xx[0]-10, 20**xx[1]-10, 20**xx[2]-10, 20**xx[3]-10])
#     fmax = 0
#     x_star = np.array([1,1,1,1])

#     return colville, transform, fmax, x_star



# def hart6():

#     alpha = np.array([1.0, 1.2, 3.0, 3.2])
#     A = np.array([[10, 3, 17, 3.5, 1.7, 8],
#                 [0.05, 10, 17, 0.1, 8, 14],
#                 [3, 3.5, 1.7, 10, 17, 8],
#                 [17, 8, 0.05, 10, 0.1, 14]])
#     P = 10**(-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
#                             [2329, 4135, 8307, 3736, 1004, 9991],
#                             [2348, 1451, 3522, 2883, 3047, 6650],
#                             [4047, 8828, 8732, 5743, 1091, 381]])
#     #hart6 = lambda x: (2.58 + np.sum([alpha[i] * np.exp(-np.sum(A[i,:] * (x - P[i,:])**2)) for i in range(4)])) / 1.94
#     hart6 = lambda xx: np.sum(alpha * np.exp(-np.sum(A * (np.array(xx) - P)**2, axis=1)))

#     # parallelizing across samples
#     # xx is of shape (n_samples, 1, 6), P is of shape (4, 6)
#     # xx - P is of shape (n_samples, 4, 6)
#     # A * (xx - P)**2 is of shape (n_samples, 4, 6)
#     # np.sum(A * (xx - P)**2, axis = 2) is of shape (n_samples, 4)
#     hart6_2 = lambda xx: np.sum(alpha * np.exp(-np.sum(A * (xx - P)**2, axis = 2)),axis=1)
#     fmax = 3.32237
#     transform = lambda xx: xx
#     x_star = None

#     return hart6, transform, fmax, x_star


# def shekel_6():


#     m = 10
#     b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
#     C = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
#                 [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
#                 [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
#                 [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])

#     shekel = lambda x: - ( (-np.sum(1 / (np.sum((x - C[:, i])**2) + b[i]) for i in range(m))) )

#     # shekel is sampled on [0,10]

#     x_star = np.array([4,4,4,4])
#     fmax = shekel(x_star)

#     transform = lambda xx: np.array([10*xx[0], 10*xx[1], 10**xx[2], 10**xx[3]])

#     return shekel, transform, fmax, x_star


# def griewank():

#     griewank = lambda x: - ( (1/4000)*np.sum(np.square(x)) - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1)))) + 1 )

#     # griewank is sampled on [-600, 600]
#     transform = lambda xx: np.array([(1200*x - 600) for x in xx])
#     fmax = 0
#     x_star = 0

#     return griewank, transform, fmax, x_star

# def rastrigin():

#     rastrigin = lambda x: - ( 10*len(x) + np.sum(np.square(x) - 10*np.cos(2*np.pi*x)) )

#     # rastrigin is sampled on [-5.12, 5.12]
#     transform = lambda xx: np.array([(10.24*x - 5.12) for x in xx])
#     fmax = 0
#     x_star = 0

#     return rastrigin, transform, fmax, x_star


# def non_zero_d():

#     # function = 1 - abs(x) - y**2

#     function = lambda x: 1 - np.abs(x[0]) - x[1]**2
#     fmax = 1
#     x_star = np.array([0,0])

#     return function, fmax, x_star


# functions for sequool

# def ridge_20_18x_2y():
#     function = lambda x: np.array(20 - 18*np.abs(x[0]) - 2*np.abs(x[1])).reshape(-1,1)
#     return function, 20, np.array([0,0]), lambda xx: xx

# def one_x2_y3(maximum = np.array([0.0,0.0])):
#     function = lambda x: np.array(1 - (x[...,0]-maximum[0])**2 - np.absolute(x[...,1]-maximum[1])**3)
#     suboptimality = lambda x: (x[...,0] - maximum[0])**2 + np.absolute(x[...,1] - maximum[1])**3
#     return function, 1, suboptimality, lambda xx: xx

# def one_x_y3(maximum = np.array([0.0,0.0])):
#     function = lambda x: np.array(1 - np.abs(x[...,0]-maximum[0]) - np.power(np.abs(x[...,1]-maximum[1]),3))
#     return function, 1, None, lambda xx: xx

# def one_x_y10(maximum = np.array([0.0,0.0])):
#     function = lambda x: np.array(1 - np.abs(x[...,0]-maximum[0]) - np.power(np.abs(x[...,1]-maximum[1]),10))
#     suboptimality = lambda x: np.abs(x[...,0] - maximum[0]) + np.power(np.abs(x[...,1] - maximum[1]),10)
#     return function, 1, suboptimality, lambda xx: xx

# def two_bumps(maximum = np.array([0.0,0.0])):
#     function = lambda x: 0.9*np.maximum(1 - (1/0.45)*np.abs(x[...,0]-0.2) - ((1/0.45)**2)*np.power(np.abs(x[...,1]-0.2),2),0) +\
#    np.maximum(1 - (1/0.45)*np.abs(x[...,0]-0.65) - ((1/0.45)**2)*np.power(np.abs(x[...,1]-0.65),2),0)
#     suboptimality = None
#     return function, 1, suboptimality, lambda xx: xx

# def one_x2_y4(maximum = np.array([0.0,0.0])):
#     function = lambda x: np.array(1 - (x[0]-maximum[0])**2 - (x[1]-maximum[1])**4)
#     suboptimality = lambda x: (x[...,0] - maximum[0])**2 + (x[...,1] - maximum[1])**4
#     return function, 1, suboptimality, lambda xx: xx

# def simple_triangle():
#     f = lambda x: np.maximum((0.2-np.absolute(x[...,0]-0.4)),0) + np.maximum((0.2-np.absolute(x[...,1]-0.4)),0)
#     return f, 0.4, None, lambda xx: xx

# def rotated_simple_triangle():
#     f = lambda x: np.maximum((0.2-np.absolute(x[...,0]-x[...,1])),0) + np.maximum((0.2-np.absolute(-x[...,0]-x[...,1]+0.8)),0)
#     return f, 0.4, None, lambda xx:xx

# def one_x_y2_high_d(maximum,angle=0):
#     d = len(maximum)
#     # thinking of x is coming from d dimension. but using only first two coordinates.
#     function = lambda x: np.array(1 - np.abs(x[...,0]*np.cos(angle) + x[...,1]*np.sin(angle)-maximum[0]) - (-x[...,0]*np.sin(angle) + x[...,1]*np.cos(angle)-maximum[1])**2)
#     suboptimality = lambda x: np.abs(x[...,0] - maximum[0]) + (x[...,1] - maximum[1])**2
#     transform = lambda xx: xx
#     fmax = 1
#     return function,fmax, suboptimality,transform

# def two_bumps_high_d_rotated(maximum = np.array([0.4/np.sqrt(2),0]),angle=np.pi/4):
    
#     function = lambda x: 1.6* np.maximum(1 - np.abs(x[...,0]*np.cos(angle) + x[...,1]*np.sin(angle)-maximum[0]) - \
#                               (-x[...,0]*np.sin(angle) + x[...,1]*np.cos(angle)-maximum[1])**2 ,0) +\
#     np.maximum(1 - (1/0.45)*np.abs(x[...,0]-0.85) - ((1/0.45)**2)*np.power(np.abs(x[...,1]-0.85),2),0)
    
#     return function, 1.6, None, lambda xx: xx

# def one_x_y2(maximum = np.array([0.0,0.0])):

#     # function = 1 - abs(x) - y**2

#     function = lambda x: np.array(1 - np.abs(x[...,0]-maximum[0]) - (x[...,1]-maximum[1])**2)
#     #function =  lambda x: np.array(1 - np.abs(x[..., 0]-0.2) - (x[..., 1]-0.2)**2)
#     #function = lambda x: np.array(1-np.abs(x[0]*np.cos(math.pi/4)-x[1]*np.sin(math.pi/4)) - (x[0]*np.sin(math.pi/4) + x[1]*np.cos(math.pi/4))**2 ).reshape(-1,1)
#     # use -1/5 to 4/5
#     #transform = lambda xx: np.array([(x-1/5) for x in xx])
#     suboptimality = lambda x: np.abs(x[...,0] - maximum[0]) + (x[...,1] - maximum[1])**2
#     transform = lambda xx: xx
#     fmax = 1
#     return function,fmax, suboptimality,transform

# def one_x2_y4_2():
    
#     function = lambda x: np.array(1 - x[0]**2 - x[1]**4).reshape(-1,1)
#     transform = lambda xx: xx
#     return function, 1, np.array([0,0]), transform
    


# def one_x_y():
#     function =  lambda x: np.array(1 - np.abs(x[..., 0]-0.5) - np.abs(x[..., 1]-0.5))
    
#     return function, 1, np.array([0.5, 0.5]), lambda xx: xx

# def one_x2_y4():
#     function = lambda x: np.array(1 - (x[..., 0]-0.5)**2 - (x[..., 1]-0.5)**4)
    
#     return function, 1, np.array([0.5, 0.5]), lambda xx: xx


    
# def one_x05_y():
#     function = lambda x: np.array(1 - np.power(np.abs(x[..., 0]-0.5),0.5) - np.abs(x[..., 1]-0.5))
    
#     return function, 1, np.array([0.5, 0.5]), lambda xx: xx
    
# def one_x_y2_2():

#     # function = 1 - abs(x) - y**2

#     #function = lambda x: np.array(1 - np.abs(x[0]-0.5) - (x[1]-0.5)**2).reshape(-1,1)
#     function =  lambda x: np.array(1 - np.abs(x[..., 0]-0.5) - (x[..., 1]-0.5)**2)
#     #function = lambda x: np.array(1-np.abs(x[0]*np.cos(math.pi/4)-x[1]*np.sin(math.pi/4)) - (x[0]*np.sin(math.pi/4) + x[1]*np.cos(math.pi/4))**2 ).reshape(-1,1)

#     # use -1/5 to 4/5
#     #transform = lambda xx: np.array([(x-1/5) for x in xx])
#     transform = lambda xx: xx
#     fmax = 1
#     x_star = np.array([0.5,0.5])
#     return function,fmax, x_star,transform


# def one_x2_y3_2():
#     # function = 1- x^2 - |y|^3
#     function = lambda x: np.array(1- np.power(x[..., 0]-0.5,2) - np.power(np.abs(x[..., 1]-0.5),3))
#     return function, 1, np.array([0.5,0.5]), lambda xx:xx
    

# def one_x2_y_2():

#     # function = 1 - abs(x) - y**2

#     #function = lambda x: np.array(1 - np.abs(x[0]-0.5) - (x[1]-0.5)**2).reshape(-1,1)
#     function =  lambda x: np.array(1 - (x[..., 0]-0.5)**2 - np.abs((x[..., 1]-0.5)))
#     #function = lambda x: np.array(1-np.abs(x[0]*np.cos(math.pi/4)-x[1]*np.sin(math.pi/4)) - (x[0]*np.sin(math.pi/4) + x[1]*np.cos(math.pi/4))**2 ).reshape(-1,1)

#     # use -1/5 to 4/5
#     #transform = lambda xx: np.array([(x-1/5) for x in xx])
#     transform = lambda xx: xx
#     fmax = 1
#     x_star = np.array([0,0])

#     return function,fmax, x_star,transform




# def one_x_y2_3():
    
#     # function = 1 - abs(x-y)/\sqrt(2) - (x+y-1)**2/2
    
#     function = lambda x: np.array(1 - np.abs(x[..., 0]-x[..., 1])/2 - (((x[..., 0]+x[..., 1]) - 1)**2)/2 )
#     transform = lambda xx: xx
#     fmax = 1
#     x_star = np.array([0,0])
    
#     return function,fmax, x_star, transform

# def one_x_y2_4():
    
#     # function is rotated 30 degrees
#     #function = 1- abs(\sqrt(3)*(x-0.5)- (y-0.5))/2 - (((x-0.5) + \sqrt(3)*(y-0.5))**2)/4
    
#     function = lambda x: np.array(1 - np.abs(np.sqrt(3)*(x[..., 0]-0.5)- (x[..., 1]-0.5))/4 - (((x[..., 0]-0.5) + np.sqrt(3)*(x[..., 1]-0.5))**2)/4 )

#     transform = lambda xx: xx
#     fmax = 1
#     x_star = np.array([0.5,0.5])
    
#     return function,fmax, x_star, transform
    


# def one_x_y2_non_orthognoal():
    
#     function = lambda x: np.array(1 - np.abs(x[..., 0]-0.5) - ((x[..., 0]+x[..., 1])/2 - 0.5)**2)
    
#     transform = lambda xx: xx
#     fmax = 1
#     x_star = np.array([0.5,0.5])
#     return function, fmax, x_star, transform

# def one_x_y2_non_orthognoal_2():
    
#     function = lambda x: np.array(1 - np.abs(x[..., 0]-0.5) - ((5*x[..., 0]+x[..., 1])/6 - 0.5)**2)
    
#     transform = lambda xx: xx
#     fmax = 1
#     x_star = np.array([0.5,0.5])
#     return function, fmax, x_star, transform


# def sum_of_ridges():
    
#     function = lambda x: np.array(1 - np.maximum(x[..., 0] - 0.5,0) - np.maximum(-(x[..., 0] - 0.5),0) \
#         - np.maximum(x[..., 0]+ x[..., 1] - 1,0)/np.sqrt(2) - np.maximum(-(x[..., 0]+ x[..., 1] - 1),0)/np.sqrt(2) ) \
            
#     return function, 1, np.array([0.5,0.5]), lambda xx: xx
    

# def y2():

#     # function = 1 - abs(x) - y**2

#     function = lambda x: np.array(1 - x[0]**2).reshape(-1,1)
#     # use -1/5 to 4/5
#     transform = lambda xx: np.array([(x-1/5) for x in xx])
#     #transform = lambda xx: xx
#     fmax = 1
#     x_star = np.array([0])

#     return function,fmax, x_star,transform

# def triangle_test():
#     #function = 1-abs(x+y)
    
#     function = lambda x: np.array(1 - np.abs(x[0]+x[1])).reshape(-1,1)
#     # use -1/5 to 4/5
#     transform = lambda xx: np.array([(x-1/5) for x in xx])
#     #transform = lambda xx: xx
#     fmax = 1
#     x_star = np.array([0,0])

#     return function,fmax, x_star,transform



#### ----------------Test Functions ------------------------- ####
## Many Local Minima. 
#GRIEWANK FUNCTION
# class Griewank():
#     def __init__(self, dim):
#         self.dim = dim
#         self.bounds =  np.vstack((np.array([-600]*self.dim),np.array([600]*self.dim)) )  # lzip([-600] * self.dim, [600] * self.dim)
#         self.min_loc = [0] * self.dim
#         self.fmin = 0
        
#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         sum1 = 0
#         prod1 = 1
#         assert self.dim == x.shape[-1]
#         d = self.dim
#         for i in range(1,d+1):
#             sum1 += x[...,i-1]**2 / 4000
#             prod1 *= np.cos(x[...,i-1] / np.sqrt(i))
#         return sum1 - prod1 + 1
            
    
# # LANGERMANN FUNCTION, gave A matrix for the 2-D case

# class Levy():
#     def __init__(self, dim):
#         self.dim = dim
#         self.bounds = np.vstack((np.array([-10]*self.dim),np.array([10]*self.dim))) #lzip([-10] * self.dim, [10] * self.dim)
#         self.min_loc = [1] * self.dim
#         self.fmin = 0
#         self.fmax = 573.929662663

#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
        
#         d = self.dim
        
#         z = [1 + (x[...,i] - 1) / 4 for i in range(d)]
        
#         sum1 = 0
#         for i in range(0,d-1):
#             sum1 += (z[i] - 1)**2 * (1 + 10 * (np.sin(np.pi * z[i] + 1))**2) 
#         return (np.sin(np.pi * z[0])**2 + sum1 + (z[d-1] - 1)**2 * (1 + np.sin(2 * np.pi * z[d-1])**2))
        
    

# class Rastrigin():
#     def __init__(self, dim=8):
#         self.dim = dim
        
#         self.bounds = np.vstack((np.array([-5.12]*self.dim),np.array([5.12]*self.dim))) #lzip([-5.12] * self.dim, [5.12] * self.dim)
#         self.min_loc = [0] * self.dim
#         self.fmin = 0
#         self.fmax = 280.61197450173

#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         d = self.dim
#         sum1 = 0
#         for i in range(1,d+1):
#             sum1 += x[...,i-1]**2 - 10 * np.cos(2 * np.pi * x[...,i-1]) 
#         return 10 * d + sum1
        
    
# class Schwefel():
#     def __init__(self,dim):
#         self.dim = dim
#         self.bounds = np.vstack((np.array([-500]*self.dim),np.array([500]*self.dim))) #lzip([-500] * self.dim, [500] * self.dim)
#         self.min_loc = [420.968746] * self.dim
#         self.fmin = 0
        

#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         sum1 = 0
#         for i in range(1,self.dim+1):
#             sum1 += x[...,i-1] * np.sin(np.sqrt(np.abs(x[...,i-1])))
        
#         return 418.982887 * self.dim - sum1
    

# # ----------------- Bowl-Shaped ------------------ #
# class Perm():
#     def __init__(self, dim,beta=10):
#         self.dim = dim
#         self.beta = beta
#         self.bounds = np.vstack((np.array([-self.dim]*self.dim),np.array([self.dim+1]*self.dim)))  #lzip([-self.dim] * self.dim, [self.dim + 1] * self.dim)
#         self.min_loc = 1 / arange(1, self.dim + 1)
#         self.fmin = 0
       
#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         outer = 0
        
#         d = self.dim
#         for i in range(1,d+1):
#             inner = 0
#             for j in range(1,d+1):
#                 inner += (j + self.beta)*(x[...,j-1]**i - (1.0/j)**i)
#             outer += inner**2
#         return outer
        

# class Rotated_Hyper_Ellipsoid():
#     def __init__(self, dim):
#         self.dim = dim
#         self.bounds = np.vstack((np.array([-65.536]*self.dim),np.array([65.536]*self.dim))) #lzip([-65.536] * self.dim, [65.536] * self.dim)
#         self.min_loc = [0] * self.dim
#         self.fmin = 0
        
#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         d = x.shape[-1]
#         outer = 0
#         for i in range(1,d+1):
#             inner = 0
#             for j in range(1,i+1):
#                 inner = inner + x[...,j-1]**2
#             outer = outer + inner
#         return outer
        
# class Sum_Of_Different_Powers():
#     def __init__(self, dim):
#         self.dim = dim
#         self.bounds =  np.vstack((np.array([-1]*self.dim),np.array([1]*self.dim))) #lzip([-1] * self.dim, [1] * self.dim)
#         self.min_loc = [0] * self.dim
#         self.fmin = 0
        
#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         d = x.shape[-1]
#         outer = np.zeros(x.shape[:-1])
#         for i in range(1,d+1):
#             outer += np.abs(x[...,i-1])**(i+1)
#         return outer
    
# class Sum_Squares():
#     def __init__(self, dim):
#         self.dim = dim
#         self.bounds =   np.vstack((np.array([-10]*self.dim),np.array([10]*self.dim))) #lzip([-10] * self.dim, [10] * self.dim)
#         self.min_loc = [0] * self.dim
#         self.fmin = 0
        
#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         d = x.shape[-1]
#         outer = np.zeros(x.shape[:-1])
#         for i in range(1,d+1):
#             outer += i* x[...,i-1]**2
#         return outer
    
# class Trid():
#     def __init__(self, dim):
#         self.dim = dim
#         self.bounds =   np.vstack((np.array([- self.dim**2]*self.dim),np.array([ self.dim**2]*self.dim)))  #lzip([-dim**2] * self.dim, [dim**2] * self.dim)
#         self.fmin = - dim*(dim+4)*(dim-1)/6
#         self.min_loc = [i*(dim+1 - i) for i in range(1,dim+1)]
        
#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         d = x.shape[-1]
#         sum1 = 0
#         for i in range(1,d+1):
#             sum1 += (x[...,i-1]-1)**2
#         sum2 = 0
#         for i in range(2, d+1):
#             sum2 += x[...,i-1]*x[...,i-2]
#         return sum1 - sum2
        
   
# ### ----------------- Plate-Shaped ------------------ ###




# ###------------------ Valley-Shaped------------------- ###
# class Three_Hump_Camel():
#     # this function is for 2D.
#     def __init__(self, dim):
#         assert dim == 2
#         self.dim = dim
#         self.bounds = np.vstack((np.array([-5]*self.dim),np.array([5]*self.dim))) 
#         self.fmin = 0
#         self.min_loc = [0]*self.dim
        
#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         return 2 * x[...,0]**2 - 1.05 * x[...,0]**4 + x[...,0]**6 / 6 + x[...,0]*x[...,1] + x[...,1]**2
    
# class Six_Hump_Camel():
#     # this function is for 2D.
#     def __init__(self, dim):
#         assert dim == 2
#         self.dim = dim
#         self.bounds = np.vstack((np.array([-3,-2]),np.array([3,2])) ) 
#         self.fmin = -1.031628453489877
#         self.min_loc = [0.0898, -0.7126]
        
#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         return (4 - 2.1 * x[...,0]**2 + x[...,0]**4 / 3) * x[...,0]**2 + x[...,0]*x[...,1] + (-4 + 4*x[...,1]**2) * x[...,1]**2
         
# class Dixon_Price():
#     def __init__(self,dim):
#         self.dim = dim
#         self.bounds = np.vstack((np.array([-10]*self.dim),np.array([10]*self.dim)))
#         self.fmin = 0
#         self.min_loc = [2**(- (2**i - 2)/2**i) for i in range(1,self.dim+1)]
#     def do_evaluate(self, x):
#         # i^th coordinate is accessed by x[...,i]
#         sum1 = 0
#         for i in range(2,self.dim+1):
#             sum1 += i * (2 * x[...,i-1]**2 - x[...,i-2])**2
#         return (x[...,0] - 1)**2 + sum1



    








    

