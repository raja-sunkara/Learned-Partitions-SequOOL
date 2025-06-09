import numpy as np
from src.utils.functions import rotation_to_original
#from shapely.geometry import Point, Polygon
import itertools
from scipy.optimize import linprog, minimize, LinearConstraint
# from scipy.spatial import ConvexHull
# import pygeos
# import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
import torch
from scipy.optimize import LinearConstraint
import cvxpy as cp
def connect_points(p1, p2,axs):
    axs.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')


# def star_cell_subopt(four_corners, f, f_star_loc=[0.5, 0.5]):
#     # find equations of the four edges
#     # eqn of a line passing through (x1, y1) and (x2, y2) is
#     # given by (y - y1)/(x - x1) = (y - y2)/(x - x2), which is
#     # x(y1 - y2) - y(x1 - x2) - x1(y1 - y2) + y1(x1 - x2) = 0
#     def line_eq(p1, p2):
#         a, b = p1 - p2
#         c = -p1[0] * b + p1[1] * a
#         return np.array([b, -a, c])

#     A = np.row_stack([
#         line_eq(four_corners[0], four_corners[1]),
#         line_eq(four_corners[1], four_corners[3]),
#         line_eq(four_corners[3], four_corners[2]),
#         line_eq(four_corners[2], four_corners[0])
#     ])
    
#     # testing the lines
#     x,y = four_corners[:,0], four_corners[:,1]
#     fig,axs = plt.subplots()
#     axs.plot(x,y, 'ro')
#     axs.scatter(f_star_loc[0], f_star_loc[1], c='g')
#     connect_points(four_corners[0], four_corners[1],axs)
#     connect_points(four_corners[1], four_corners[3],axs)
#     connect_points(four_corners[3], four_corners[2],axs)
#     connect_points(four_corners[2], four_corners[0],axs)
#     fig.savefig('four_corners.png')

#     plt.close()
    
#     # the four corners are arranged before the rotation as
#     # 0 : bot left, 1 : top left, 2: bot right, 3: top right
#     midpt = np.mean(four_corners, axis=0)
#     midpt_signs = np.sign(A @ np.concatenate((midpt, np.ones(1))))
#     interior = LinearConstraint(midpt_signs.reshape(-1, 1) * A, lb = 0)
#     # third_coord = LinearConstraint(np.array([0, 0, 1]), lb = 1, ub = 1)

#     def obj_func(x):
#         return f(x[:-1]) # third coord added to variable for constraint eq

#     return minimize(
#         obj_func,
#         np.concatenate((midpt, np.ones(1))),
#         constraints = interior, #third_coord],
#         bounds = [(np.min(four_corners[:,0]), np.max(four_corners[:,0])), (np.min(four_corners[:,1]), np.max(four_corners[:,1])), (1, 1)]        
#     )

def intersect_unit_square(m, x0=0.5, y0=0.5):
    # Calculate the y-intercept of the line
    b = y0 - m * x0

    # Find the intersection points of the line with the unit square
    points = []
    for x, y in [(0, b), (1, m + b), ((1 - b) / m, 1), (-b / m, 0)]:
        if 0 <= x <= 1 and 0 <= y <= 1:
            points.append((float(x), float(y)))

    return points


def rotation_matrix(w):

    # define the direction vector w
    # w = np.array([1,1,0])

    # normalize the direction vector to obtain a unit vector u
    u = w / np.linalg.norm(w)

    # construct an orthonormal basis for the d-dimensional space
    d = len(w)
    v = np.zeros((d, d-1))
    v[:,0] = np.random.randn(d)
    v[:,0] = v[:,0] - np.dot(u,v[:,0])*u
    v[:,0] = v[:,0] / np.linalg.norm(v[:,0])
    for i in range(1,d-1):
        v[:,i] = np.random.randn(d)
        for j in range(i):
            v[:,i] = v[:,i] - np.dot(v[:,j],v[:,i])*v[:,j]
        v[:,i] = v[:,i] - np.dot(u,v[:,i])*u
        v[:,i] = v[:,i] / np.linalg.norm(v[:,i])

    # construct the basis matrix and its inverse
    B = np.column_stack((u, v))
    B_inv = np.linalg.inv(B)

    # construct the rotation matrix that transforms the basis into the standard basis
    I = np.eye(d)
    R = B_inv.dot(I)

    # apply the rotation matrix to the unit vectors e1, e2, e3, etc.
    # e1 = np.zeros(d)
    # e1[0] = 1
    # e = np.zeros((d,d))
    # for i in range(d):
    #     e[i,:] = R.dot(I[:,i])

    # # print the transformed basis vectors
    # print(e)

    return R

def transformed_box_extent(R):
    d = len(R)
    corners = np.zeros((2**d, d))
    for i in range(d):
        mask = 2**i
        corners[:,i] = np.repeat([0,1], 2**i).tolist() * (2**(d-i-1))

    # apply the rotation matrix to each corner of the unit cube
    corners = corners - np.array([0.5]*d)
    corners_rot = corners.dot(R.T)
    # corners_rot = np.dot(R.T,corners.T)

    # find the minimum and maximum coordinates of the transformed cube in each dimension
    min_coords = corners_rot.min(axis=0)
    max_coords = corners_rot.max(axis=0)


    return min_coords, max_coords

def two_ridges(W1):

    # W = np.array([[3],[1],[-0.6]])
    W = W1[:,0]
    slope1 = W[1]/W[0]
    theta1 = np.arctan(slope1)
    R1 = np.array([[np.cos(theta1), np.sin(theta1)], [-np.sin(theta1), np.cos(theta1)]]).reshape(2,2)
    points1 = intersect_unit_square(slope1)

    W = W1[:,1]
    slope2 = W[1]/W[0]
    theta2 = np.arctan(W[1]/W[0])
    R2 = np.array([[np.cos(theta2), np.sin(theta2)], [-np.sin(theta2), np.cos(theta2)]])
    points2 = intersect_unit_square(slope2)

    corners = np.array([[0,0],[1,0],[0,1],[1,1]])
    # Apply the rotation
    p_rotated1 = np.dot(R1, (corners-[0.5,0.5]).T)
    p_rotated2 = np.dot(R2, (corners-[0.5,0.5]).T)

    x_max1,y_max1 = np.max(p_rotated1,axis=1)
    x_min1,y_min1 = np.min(p_rotated1,axis=1)
    # thse four acts as a limits now.

    x_max2,y_max2 = np.max(p_rotated2,axis=1)
    x_min2,y_min2 = np.min(p_rotated2,axis=1)

    return points1,points2, R1, R2, [x_min1,y_min1, x_max1, y_max1], [x_min2, y_min2,x_max2,y_max2]


def find_intersection(w):
    # Find the slope of the line defined by w[0]*x + w[1]*y + w[2]
    slope = -w[0] / w[1]

    # Find the slope of the perpendicular line passing through (0.5, 0.5)
    perp_slope = -1 / slope

    # Solve the system of equations to find the intersection point
    x = (-w[2] - w[1]*0.5*(1-perp_slope))/(w[0]+w[1]*perp_slope)
    y = perp_slope*(x - 0.5) + 0.5

    return x, y

def point_on_line(left, right, i_star):
    # Compute the slope and y-intercept of the line defined by left and right
    m = (right[1] - left[1]) / (right[0] - left[0])
    b = left[1] - m*left[0]

    # Compute the y-coordinate of i_star on the line defined by left and right
    y_i_star = m*i_star[0] + b

    # Check if i_star lies on the line segment defined by left and right
    if (left[1] <= y_i_star <= right[1] or right[1] <= y_i_star <= left[1]) and \
            (left[0] <= i_star[0] <= right[0] or right[0] <= i_star[0] <= left[0]):
        return True
    else:
        return False

def point_in_square(min_point, max_point, test_point):
    """
    Check if a point is inside a rectangle defined by its minimum and maximum coordinates.

    Args:
    min_point (tuple): The minimum x and y coordinates of the rectangle as a tuple (min_x, min_y).
    max_point (tuple): The maximum x and y coordinates of the rectangle as a tuple (max_x, max_y).
    test_point (tuple): The x and y coordinates of the point to test as a tuple (test_x, test_y).

    Returns:
    bool: True if the test point is inside the rectangle, False otherwise.
    """
    return (min_point[0] <= test_point[0] <= max_point[0]) and (min_point[1] <= test_point[1] <= max_point[1])

def sample_random_points(num_samples,d,f,rng, cell = None,**kwargs):
    
    bounds = kwargs.get('bounds', None)
    if bounds is not None:
        min_point = bounds[0]
        max_point = bounds[1]
    else:
        min_point = np.min(cell, axis=0)
        max_point = np.max(cell, axis=0)
        
    
    #if cell is None:
    #    cell = np.array(list(itertools.product([0,1], repeat=d)))    
    # what happens for the general d. cell is a 2^d x d matrix
   
    
    # np.random.uniform can take d-dimensional vectors as input
    data_x = rng.uniform(min_point,max_point,(num_samples,d))
    
    # sample from the normal distirbution canterd at the center of the cell
    #data_x = rng.normal(0,1,(num_samples,d))
    
       
    #x_min, x_max, y_min, y_max = np.amin(cell[:,0]), np.amax(cell[:,0]), np.amin(cell[:,1]), np.amax(cell[:,1])
    #u1 = np.random.uniform(x_min,x_max,num_samples)
    #u2 = np.random.uniform(y_min,y_max,num_samples)
    
    #data_x = np.stack((u1,u2),axis=-1)
    #data_x = np.random.uniform(min,max,(num_samples,d))
    
    data_y = f(data_x)
    X = torch.tensor(data_x).type(torch.float64)
    Y = torch.tensor(data_y).type(torch.float64).reshape(-1,1)
    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)
    
    
    
    return X,Y


# def point_in_parallelogram(point1, point2, point3, point4, test_point):
    
#     parallelogram = Polygon([point1, point2, point3, point4,point1])
#     test_point = Point(test_point)
#     return parallelogram.contains(test_point) or parallelogram.within(test_point)
# #    return parallelogram.within(test_point)

def point_in_parallelogram_high_dim(four_corners, test_point):
    # this function checks if the test_point is in the parallelogram former by point1, point2, point3, point4
    # point1, point2, point3, point4 are 4 points in high dimension
    # test_point is a point in high dimension

    first_edge = four_corners[1] - four_corners[0]
    second_edge = four_corners[3] - four_corners[0]

    perp_first_edge = second_edge - np.dot(second_edge, first_edge) * first_edge / np.linalg.norm(first_edge)
    perp_second_edge = first_edge - np.dot(first_edge, second_edge) * second_edge / np.linalg.norm(second_edge)

    # any point x on first_edge satisfies np.dot(x - four_corners[0], perp_first_edge) == 0
    # any point on other edge parallel to first_edge satisfies np.dot(x - four_corners[3], perp_first_edge) == 0

    b_first_edge = np.dot(four_corners[0], perp_first_edge)
    b_parallel_first_edge = np.dot(four_corners[3], perp_first_edge)

    # any point x on ssecond_edge satisfies np.dot(x - four_corners[0], perp_second_edge) == 0

    b_second_edge = np.dot(four_corners[0], perp_second_edge)
    b_parallel_second_edge = np.dot(four_corners[1], perp_second_edge)

#    test_pt = np.average(four_corners, axis=0)
    test_pt = test_point


    return ((np.dot(test_pt, perp_first_edge) - b_first_edge, np.dot(test_pt, perp_first_edge) - b_parallel_first_edge), (np.dot(test_pt, perp_second_edge) - b_second_edge, np.dot(test_pt, perp_second_edge) - b_parallel_second_edge))



def area_of_triangle(point_a,point_b,point_c):
    # a, b, c are the vertices of the triangle
    a = np.linalg.norm(point_b - point_c)
    b = np.linalg.norm(point_a - point_c)
    c = np.linalg.norm(point_a - point_b)

    s = (a + b + c) / 2 
    area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    return area

def point_inside_parallelogram(point0,point1,point2,point3,test):
    # point1, point2, point3, point4 are the vertices of the parallelogram
    # test is the point to be tested
    # point0 is the origin of the coordinate system

    traingle_area = area_of_triangle(point0, point1, test) + area_of_triangle(point1, point2, test) + area_of_triangle(point2, point3, test) + area_of_triangle(point0, point3, test) 
    
    parallelogram_area = area_of_triangle(point0, point1, point2) + area_of_triangle(point0, point3,point2)

    print(traingle_area, parallelogram_area)

    if abs(traingle_area - parallelogram_area) < 10:
        return True
    else:
        return False


def split_parallelogram(points):
    # Find the length of the x and y axes

    x_length = np.max(points[0]) - np.min(points[0])
    y_length = np.max(points[1]) - np.min(points[1])

    # Determine the largest axis
    if x_length >= y_length:
        axis = 0
    else:
        axis = 1

    # Sort the points along the largest axis
    sorted_points = points[:, np.argsort(points[axis])]

    # Calculate the splitting points along the largest axis
    dx = (sorted_points[axis][-1] - sorted_points[axis][0]) / 3
    x1 = sorted_points[axis][0] + dx
    x2 = sorted_points[axis][0] + (2 * dx)

    # Split the parallelogram into three parts
    p1 = sorted_points[:, :2]
    p2 = np.array([sorted_points[:, 1], sorted_points[:, 2], [x1, sorted_points[1, 2]], [x1, sorted_points[1, 1]]])
    p3 = np.array([[x2, sorted_points[1, 2]], [x2, sorted_points[1, 1]], sorted_points[:, 2:], sorted_points[:, :1]])

    # Sort the points back to their original order
    p1 = p1[:, np.argsort(sorted_points[axis][:2])]
    p2 = p2[:, np.argsort(np.concatenate((sorted_points[axis][1:3], [x1])))]
    p3 = p3[:, np.argsort(np.concatenate(([x2], sorted_points[axis][2:], sorted_points[axis][:1])))]

    # Return the coordinates of the three parts
    return p1, p2, p3



def find_longest_axis(points):
    # Extract the x and y coordinates of the parallelogram points
    # points = parallelogram['parallelogram']
    x_coords = points[0]
    y_coords = points[1]

    # Calculate the length of the x and y axes
    x_length = np.max(x_coords) - np.min(x_coords)
    y_length = np.max(y_coords) - np.min(y_coords)

    # Check which axis has the larger length
    if x_length >= y_length:
        return 0  # x-axis has the largest length
    else:
        return 1  # y-axis has the largest length

def representative_point(x_min_u, x_max_u, R1, R2):

    # calculate the remaining two points
    p1 = np.array([x_min_u[0], x_max_u[1]])
    p2 = np.array([x_max_u[0], x_min_u[1]])

    # converting these to global frame
    x_min_g_1 = rotation_to_original(R1.T,x_min_u)
    x_max_g_2 = rotation_to_original(R1.T,x_max_u)
    x_min_g_3 = rotation_to_original(R1.T,p1)
    x_max_g_4 = rotation_to_original(R1.T,p2)
    
    # convert these points to v frame
    x_v_1 = np.dot(R2, (x_min_g_1-[0.5,0.5]).T)
    x_v_2 = np.dot(R2, (x_max_g_2-[0.5,0.5]).T)
    x_v_3 = np.dot(R2, (x_min_g_3-[0.5,0.5]).T)
    x_v_4 = np.dot(R2, (x_max_g_4-[0.5,0.5]).T)

    # find the extent of the box 
    points = np.hstack((x_v_1,x_v_2,x_v_3,x_v_4))
    x_max,y_max = np.max(points,axis=1)
    x_min,y_min = np.min(points,axis=1)

    # v = constant line: (x_max+x_min)/2
    v = (x_max + x_min)/2

    return v

def u_to_v(x_min_u, x_max_u, R1, R2):

        # calculate the remaining two points
    p1 = np.array([x_min_u[0], x_max_u[1]])
    p2 = np.array([x_max_u[0], x_min_u[1]])

    # converting these to global frame
    x_min_g_1 = rotation_to_original(R1.T,x_min_u)
    x_max_g_2 = rotation_to_original(R1.T,x_max_u)
    x_min_g_3 = rotation_to_original(R1.T,p1)
    x_max_g_4 = rotation_to_original(R1.T,p2)
    
    # convert these points to v frame
    x_v_1 = np.dot(R2, (x_min_g_1-[0.5,0.5]).T)
    x_v_2 = np.dot(R2, (x_max_g_2-[0.5,0.5]).T)
    x_v_3 = np.dot(R2, (x_min_g_3-[0.5,0.5]).T)
    x_v_4 = np.dot(R2, (x_max_g_4-[0.5,0.5]).T)

    # Find the minimum and maximum x and y coordinates
    x_coords = [p[0] for p in [x_v_1, x_v_2, x_v_3, x_v_4]]
    y_coords = [p[1] for p in [x_v_1, x_v_2, x_v_3, x_v_4]]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Define the two corner points using the minimum and maximum coordinates
    x_min_y_min = np.array([x_min, y_min])
    x_max_y_max = np.array([x_max, y_max])

    return x_min_y_min, x_max_y_max


def parallelogram_points(A,d):

    alpha_beeta = []
    for corner in itertools.product(range(2), repeat=d):
        alpha_beeta.append(np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),np.array(corner)))
    alpha_beeta = np.array(alpha_beeta).T

    # computing the extent corners
    alpha_extent = [np.min(alpha_beeta[0]), np.max(alpha_beeta[0])]
    beeta_extent = [np.min(alpha_beeta[1]), np.max(alpha_beeta[1])]

    parallelogram_corners = []
    for a in alpha_extent:
        for b in beeta_extent:
            parallelogram_corners.append(a * A[:,0] + b * A[:,1])

    parallelogram_corners[2], parallelogram_corners[3] = parallelogram_corners[3], parallelogram_corners[2]

    return parallelogram_corners


# def parallelogram_points_d(A,d):


#     alphas = []
#     for corner in itertools.product(range(2), repeat=d):
#         alphas.append(np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),np.array(corner)))
#     alphas = np.array(alphas).T

#     # computing the extent corners
#     extent = np.vstack((np.min(alphas,axis=1),np.max(alphas,axis=1)))
#     two_power_d = list(itertools.product(*extent.T))
#     two_power_d_corners = A @ np.array(two_power_d).T

#     return two_power_d_corners.T

# def labelled_corners2(A,angle):
#     two_power_d_corners = []
#     d,p = A.shape[0],A.shape[1]
#     rot_matrix = np.array([[np.cos(angle), -1 * np.sin(angle)],
#                            [np.sin(angle), np.cos(angle)]])
#     for corner in itertools.product(range(2), repeat=d):
#         two_power_d_corners.append(rot_matrix @ np.array(corner).T)
#     two_power_d_corners = np.array(two_power_d_corners)
#     binary = list(itertools.product(range(2), repeat=p))
#     labeled_points = [(point,label) for point, label in zip(two_power_d_corners.T,binary)]
    
#     return labeled_points
     
        
    

def labelled_corners(A,initial_domain):
    '''
    A: the columns of the matrix represent the directions (each directions is on column). and there are d directions.
    Output: The extent of the hypercube in the d-dimensional space.
    Along with the labels of the corners.(Binary representation of the corners)
    
    initial_dimain: is a 2 x d matrix. The first row represents the minimum extent of the hypercube i
    in each dimension. And the second row maximum extent.
    '''
    if initial_domain is None:
        domain = [(0,1) for i in range(A.shape[0])]
    else:
        # making the array as a list of tuples.
        domain = [(initial_domain[0,i],initial_domain[1,i]) for i in range(A.shape[0])]
    
    alphas = []
    d,p = A.shape[0],A.shape[1]
    
    if d >= 15:
        #points = np.random.uniform(initial_domain[0],initial_domain[1],(200000,d))
        points = np.random.default_rng(42).uniform(1*initial_domain[0],1*initial_domain[1],(400000,d))
    else:
        points = np.array(list(itertools.product(*domain)))
 
    # for corner in points: #itertools.product(range(2), repeat=d):
    #     alphas.append(np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),np.array(corner)))
    #     #alphas.append(A.T @ np.array(corner))

    alphas = (A.T @ points.T)  
        
    # in high-d this for loop takes lot of time. so unifromly samples 100000 k points on the domain
    # and then computes the extent of the hypercube.
    
    
#    alphas = np.array(alphas).T

    # computing the extent corners
    extent = np.vstack((np.min(alphas,axis=1),np.max(alphas,axis=1)))
    #two_power_d = list(itertools.product(*extent.T))
    #binary = list(itertools.product(range(2), repeat=p))
    #two_power_d_corners = A @ np.array(two_power_d).T

    #labeled_points = [(point,label) for point, label in zip(two_power_d_corners.T,binary)]

    return None, extent


def group_labelled_corners(labeled_points,p):
    """
    Input: The labeled extent corners of the hypercube.
    Output: Grouping all the pair of points that are parallel to each direction.
    """
    # grouping points by parallel directions
    # list consitinf of p lists

    grouped_points = [[] for i in range(p)]
    for dim in range(p):
        grouped_points[dim] = []
        for i in range(len(labeled_points)):
            for j in range(i+1, len(labeled_points)):
                label1, label2 = labeled_points[i][1], labeled_points[j][1]
                if label1[dim] != label2[dim] and sum([label1[k] != label2[k] for k in range(p)]) == 1:
                    parallel_points = [labeled_points[i][0], labeled_points[j][0]]
                    grouped_points[dim].append((parallel_points)) 

    return np.array(grouped_points)

class X_scaler:
    def __init__(self,maximum,cell):
        self.maximum = maximum
        self.epsilon = np.amax(cell,axis=0) - np.amin(cell,axis=0)
        
    def fit_transform(self,X):
        return ((X-self.maximum)/torch.tensor(self.epsilon))

class X_scaler_high_d:
    def __init__(self, cell):
        self.center = np.mean(cell,axis=0)
        self.epsilon = np.amax(cell,axis=0) - np.amin(cell,axis=0)
    def fit_transform(self,X):
        return ((X-self.center)/torch.tensor(self.epsilon))
    
class Y_scaler:
    def __init__(self):
        self.y_min = None
        self.y_max = None
    def fit_transform(self,Y):
        self.y_min = min(Y)
        self.y_max = max(Y)
        return (Y-self.y_min)/(self.y_max-self.y_min)
    def inverse_transform(self,Y):
        return Y* ((self.y_max-self.y_min).item()) + (self.y_min).item()
    
def top_k_svd_directions(X,reduction=True,variance=0.95,M=None):
    # Compute the SVD of the data matrix X
    U, S, V = np.linalg.svd(X, full_matrices=False)

    # Calculate the percentage of variance explained by each principal component
    total_variance = np.sum(S**2)
    variance_explained = S**2 / total_variance

    # Determine the number of principal components to retain
    cumulative_variance = np.cumsum(variance_explained)
    n_components = np.argmax(cumulative_variance >= variance) + 1 # Retain enough to explain 95% of variance
    if M:
        n_components = M
    # Construct the reduced SVD
    U_reduced = U[:, :n_components] 
    S_reduced = np.diag(S[:n_components])
    V_reduced = V[:n_components, :]
    
    if reduction:
        return U_reduced, S_reduced, V_reduced, U
    else:
        return U, S, V, U   

def project_onto_boundary(A, ortho_normal_basis, boundary, point):
    # if the point is outside the boundary, project it onto the boundary
    # which is orthogonal to the subspace spanned by the columns of A
    
    # if the point is inside the boundary, return the point
    if np.all(np.logical_and(boundary[0] <= point, point <= boundary[1])):
        return point, 1
    # point is outside the boundary,
    
    elif A.shape[0] == A.shape[1]:
        # no reduction in the search space
        return point, 0
    elif A.shape[0] > A.shape[1]:   # remaining vectors are in ortho_normal_basis
        # feasisible point check optimization.
        # point is on the subspace A, need to find orthogonal projection on the boundary.
        A_prime = ortho_normal_basis[:,A.shape[1]:]
        def distance(x):
            # x is a numpy array
            return np.linalg.norm(A_prime @ x)
        # constrains: (point + A_prime @ x) should belong to boundary. which is
        # boundary[0] <= point + A_prime @ x <= boundary[1]. This can be written as
        # boundary[0] - point <= A_prime @ x <= boundary[1] - point
        constraint = LinearConstraint(A_prime, boundary[0] - point, boundary[1] - point)
        res = minimize(distance, constraints=[constraint],options={'verbose': 1})
        print(res.x, point+ A_prime @ res.x)
   
def dantzig_selector(f, d, rng, logger,epsilon = 1e-5, m_phi = 5, m_x = 10):  
       
    X = rng.standard_normal((m_x,d))   # (m_x \times d)
    X = X/np.linalg.norm(X, axis=1,keepdims=True)
    
    # sample phi
    
    
    def generate_phi_with_rank(m_phi, dimension, rng):
        count = 0
        while True:
            phi = rng.choice([-1, 1], size=(m_phi, dimension)) / np.sqrt(m_phi)
            count +=1
            if matrix_rank(phi) == m_phi:
                return phi, count

    #phi =  rng.choice([-1, 1], size = (m_phi,d))  / np.sqrt(m_phi)        # it is (m_phi \times d) matrix
    # Y_ij = (f(X[j] + epsilon phi[i]) - f(X[j]))/epsilon
    phi, count = generate_phi_with_rank(m_phi, d, rng)
    logger.info(f"samples phi matrix, {count} times to ensure rank(phi) = row_rank")
    logger.info(f"rank of phi is {matrix_rank(phi)}")
    
    Y = np.zeros((m_phi,m_x))

    function_values = []
    for i in range(m_phi):
        for j in range(m_x):
            #print((X[j] + epsilon * phi[j]))
            f_x1 = f(X[j] + epsilon * phi[i])
            f_x2 = f(X[j])
            function_values.append(f_x1)
            function_values.append(f_x2)
            Y[i,j] = (f_x1 - f_x2)/epsilon
    # X[:,None,:] + epsilon * phi[None,:,:]    # this is of shape (m_x, m_phi, d)
    assert Y.shape == (m_phi,m_x) and X.shape == (m_x,d) and phi.shape == (m_phi,d)
    
    X_hat = []
    for j in range(m_x):
        x = cp.Variable(d) # Variable to optimize
        objective = cp.Minimize(cp.norm(x, 1))
        constraints = [phi @ x == Y[:,j]]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=False)

        # print("Optimal value: ", result)
        X_hat.append(x.value)
    X_hat = np.array(X_hat).T

    #assert X_hat.shape == (d,m_x)
    if X_hat.shape != (d,m_x):
        return False, False, False
    U,S,V = np.linalg.svd(X_hat.T)
    return V, function_values,True,S

