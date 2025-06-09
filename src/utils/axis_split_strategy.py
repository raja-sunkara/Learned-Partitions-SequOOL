import numpy as np
import matplotlib.pyplot as plt
from utils.rotations import point_in_parallelogram

def function_minimum(f, cell):
    t1 = np.linspace(0, 1, 100)
    t2 = np.linspace(0, 1, 100)
        
    start_point = cell[0] + t2[:, np.newaxis] * (cell[2] - cell[0])
    end_point = cell[1] + t2[:, np.newaxis] * (cell[3] - cell[1])
        
    # Generate points on line segments between start and end points
    points_along_axis1 = []
    for i in range(100):
        points_along_axis1.append(start_point[i] + t1[:, np.newaxis] * (end_point[i] - start_point[i]))
    points_along_axis1 = np.array(points_along_axis1)  # points is of shape (100, 100, d)
    fig,axs = plt.subplots(1,1)
    img = axs.scatter(points_along_axis1[:,:,0],points_along_axis1[:,:,1], c= np.squeeze(f(points_along_axis1)))
    plt.colorbar(img,ax=axs) 
    fig.savefig('all_points.png')
    
    # take aveage of least 10 values
    
    return np.median(np.sort(np.squeeze(f(points_along_axis1)).reshape(-1))[:100])
    
    # def area_average():
    #     nn_vals = np.squeeze(f(points_along_axis1))
    #     threshold = np.amin(nn_vals) + 0.1*(np.amax(nn_vals) - np.amin(nn_vals))
    #     return np.sum(nn_vals <= threshold)
    
    # return area_average()
    #return np.amin(np.squeeze(f(points_along_axis1)))
    #return np.amin(np.squeeze(f(points_along_axis1)))
    


def function_variation_computation(cell,function_to_optimize2):
    t1 = np.linspace(0, 1, 50)
    t2 = np.linspace(0, 1, 50)
    
    start_point = cell[0] + t2[:, np.newaxis] * (cell[2] - cell[0])
    end_point = cell[1] + t2[:, np.newaxis] * (cell[3] - cell[1])
    
    # Generate points on line segments between start and end points
    points_in_rectangle = []
    for i in range(50):
        points_in_rectangle.append(start_point[i] + t1[:, np.newaxis] * (end_point[i] - start_point[i]))
    points_in_rectangle = np.array(points_in_rectangle)
    #points_along_axis1 = start_point[:, :, np.newaxis] + t_values[:, np.newaxis, np.newaxis] * (end_point[:, np.newaxis, :] - start_point[:, :, np.newaxis])
    
    
    nn_vals = np.squeeze(function_to_optimize2(points_in_rectangle))
    
    if nn_vals.shape[0] == 50*50:
        nn_vals = nn_vals.reshape(50,50)
    
    axis1_variation = np.sum(np.abs(np.diff(nn_vals,axis=1)))
    
    fig, axs = plt.subplots(1,1,figsize=(5,5))
    img = axs.scatter(points_in_rectangle[:,:,0],points_in_rectangle[:,:,1], c = np.squeeze(function_to_optimize2(points_in_rectangle)))
    plt.colorbar(img,ax=axs)
    fig.savefig('all_cells_function_variation.png')

    axis2_variation = np.sum(np.abs(np.diff(nn_vals,axis=0)))
    return axis1_variation,axis2_variation

# def function_variation_computation2(cell, f_nn,nn_model,svd_directions):
#     t1 = np.linspace(0, 1, 100)
#     t2 = np.linspace(0, 1, 100)
    
#     start_point = cell[0] + t2[:, np.newaxis] * (cell[2] - cell[0])
#     end_point = cell[1] + t2[:, np.newaxis] * (cell[3] - cell[1])
    
#     # Generate points on line segments between start and end points
#     points_along_axis1 = []
#     for i in range(100):
#         points_along_axis1.append(start_point[i] + t1[:, np.newaxis] * (end_point[i] - start_point[i]))
#     points_along_axis1 = np.array(points_along_axis1)  # points is of shape (100, 100, d)
    
#     # get neaural network weights and biases.
#     directions = nn_model.linear_one.weight.to('cpu').detach().numpy().T  # directions is of shape (d, n_hidden)
#     biases = nn_model.linear_one.bias.to('cpu').detach().numpy().T       # biases is of shape (n_hidden,)
#     second_layer_weights = nn_model.linear_two.weight.to('cpu').detach().numpy().T  # second layer weights is of shape (n_hidden, 1)
    
#     activated_neurons = (points_along_axis1 @ directions + biases >= 0)   # is of shape (100, 100, n_hidden)
#     weighted_directions = directions * second_layer_weights.T  # is of shape (2, n_hidden)
    
#     gradients = activated_neurons @ weighted_directions.T  # is of shape (100, 100, 2)
#     axis_1_directional_derivative = gradients @ svd_directions[:,0]
#     axis_2_directional_derivative = gradients @ svd_directions[:,1]
    
#     axis1_variation = np.sum(np.abs(np.diff(axis_1_directional_derivative,axis=1)))
#     axis2_variation = np.sum(np.abs(np.diff(axis_2_directional_derivative,axis=0)))
    
    
#     return axis1_variation,axis2_variation


# def function_variation_computation3(cell, f_nn,nn_model,svd_directions):
#     # compute the gradient at the center
#     representative_point = np.mean(cell,axis=0,keepdims=True)  # is of shape (1, d)

#     # get neaural network weights and biases.
#     directions = nn_model.linear_one.weight.to('cpu').detach().numpy().T  # directions is of shape (d, n_hidden)
#     biases = nn_model.linear_one.bias.to('cpu').detach().numpy().T       # biases is of shape (n_hidden,)
#     second_layer_weights = nn_model.linear_two.weight.to('cpu').detach().numpy().T  # second layer weights is of shape (n_hidden, 1)
    
#     activated_neurons = (representative_point @ directions + biases >= 0)   # is of shape (1, n_hidden)
#     weighted_directions = directions * second_layer_weights.T  # is of shape (2, n_hidden)
    
#     gradient = activated_neurons @ weighted_directions.T  # is of shape (1, 2)
#     axis_1_directional_derivative = np.absolute(gradient @ svd_directions[:,0])
#     axis_2_directional_derivative = np.absolute(gradient @ svd_directions[:,1])
    
#     return axis_1_directional_derivative,axis_2_directional_derivative

# def generate_points(i_star_cell):
#     t1 = np.linspace(0, 1, 200)
#     t2 = np.linspace(0, 1, 200)

#     start_point = i_star_cell[0] + t2[:, np.newaxis] * (i_star_cell[2] - i_star_cell[0])
#     end_point = i_star_cell[1] + t2[:, np.newaxis] * (i_star_cell[3] - i_star_cell[1])
        
#     # Generate points on line segments between start and end points
#     points_along_axis1 = []
#     for i in range(200):
#         points_along_axis1.append(start_point[i] + t1[:, np.newaxis] * (end_point[i] - start_point[i]))
#     points_along_axis1 = np.array(points_along_axis1)  # points is of shape (100, 100, d)
#     return points_along_axis1

# we need to have choose_axis_to_split_minimum_approach for the general case (d >2)
# def choose_axis_to_split_minimum_approach(i_star_cell, fun,maximum,h):
#     '''
#     i_star_cell: the cell to be split. will be of shape (2^d,d).
#     fun: neural network function.
#     '''
#     # write with for loop now, later we can vectorize it
#     for axis in range(i_star_cell.shape[1]):
        
        



def choose_axis_to_split_minimum_approach(i_star_cell, fun,maximum,h):
    i_star_cell_found = 0
    left_cell = i_star_cell.copy()
    left_cell[1] = i_star_cell[0] + (i_star_cell[1]-i_star_cell[0])/3.0
    left_cell[3] = i_star_cell[2] + (i_star_cell[3]-i_star_cell[2])/3.0
    i_star_h = point_in_parallelogram(left_cell[0],left_cell[1],left_cell[3],left_cell[2],maximum)
    if i_star_h:
        print(f"i star cell found at depth {h}")
        x_division_star_cell = left_cell.copy()
        i_star_cell_found +=1
    right_cell = i_star_cell.copy()
    right_cell[0] = i_star_cell[1] + (i_star_cell[0]-i_star_cell[1])/3.0
    right_cell[2] = i_star_cell[3] + (i_star_cell[2]-i_star_cell[3])/3.0
    i_star_h = point_in_parallelogram(right_cell[0],right_cell[1],right_cell[3],right_cell[2],maximum)
    if i_star_h:
        print(f"i star cell found at depth {h}")
        x_division_star_cell = right_cell.copy()
        i_star_cell_found +=1
    middle_cell = i_star_cell.copy()
    middle_cell[0] = left_cell[1]       
    middle_cell[1] = right_cell[0]
    middle_cell[2] = left_cell[3]        
    middle_cell[3] = right_cell[2]
    i_star_h = point_in_parallelogram(middle_cell[0],middle_cell[1],middle_cell[3],middle_cell[2],maximum)
    if i_star_h:
        print(f"i star cell found at depth {h}")
        x_division_star_cell = middle_cell.copy()
        i_star_cell_found +=1
    #sub_optimality_after_x_division = f_hat_star - star_cell_subopt(x_division_star_cell,function_to_optimize2,maximum).fun
    sub_optimality_after_x_division = function_minimum(fun,x_division_star_cell)
    #sub_optimality_after_x_division = star_cell_subopt(x_division_star_cell,function_to_optimize2,maximum).fun
    #min_location1 = star_cell_subopt(x_division_star_cell,function_to_optimize2,maximum).x
    # for suboptimality, evaluate at the four corners
    #sub_optimality_after_x_division = np.amin(fun(x_division_star_cell))
    
    #sanity check
    #point_in_parallelogram(x_division_star_cell[0],x_division_star_cell[1],x_division_star_cell[3],x_division_star_cell[2],
    #                       sub_optimality_after_x_division.x[:-1])
    
    # suppose we decide to split along axis = 1
    bottom_cell = i_star_cell.copy()
    bottom_cell[3] = i_star_cell[1] + (i_star_cell[3] - i_star_cell[1])/3.0
    bottom_cell[2] = i_star_cell[0] + (i_star_cell[2] - i_star_cell[0])/3.0
    i_star_h = point_in_parallelogram(bottom_cell[0],bottom_cell[1],bottom_cell[3],bottom_cell[2],maximum)
    if i_star_h:
        print(f"i star cell found at depth {h}")
        y_division_star_cell = bottom_cell.copy()
        i_star_cell_found +=1

    top_cell = i_star_cell.copy()
    top_cell[0] = i_star_cell[2] + (i_star_cell[0] - i_star_cell[2])/3.0
    top_cell[1] = i_star_cell[3] + (i_star_cell[1] - i_star_cell[3])/3.0
    i_star_h = point_in_parallelogram(top_cell[0],top_cell[1],top_cell[3],top_cell[2],maximum)
    if i_star_h:
        print(f"i star cell found at depth {h}")
        y_division_star_cell = top_cell.copy()
        i_star_cell_found +=1

    middle_cell= i_star_cell.copy()
    middle_cell[0] = bottom_cell[2]
    middle_cell[1] = bottom_cell[3]
    middle_cell[2] = top_cell[0]
    middle_cell[3] = top_cell[1]
    i_star_h = point_in_parallelogram(middle_cell[0],middle_cell[1],middle_cell[3],middle_cell[2],maximum)
    if i_star_h:
        print(f"i star cell found at depth {h}")
        y_division_star_cell = middle_cell.copy()
        i_star_cell_found +=1
        
    #sub_optimality_after_y_division = f_hat_star - star_cell_subopt(y_division_star_cell,function_to_optimize2,maximum).fun
    #sub_optimality_after_y_division = star_cell_subopt(y_division_star_cell,function_to_optimize2,maximum).fun
    sub_optimality_after_y_division = function_minimum(fun,y_division_star_cell)
    #min_location2 = star_cell_subopt(y_division_star_cell,function_to_optimize2,maximum).x
    #sub_optimality_after_y_division =  np.amin(fun(y_division_star_cell))
    # if np.allclose(sub_optimality_after_x_division,sub_optimality_after_y_division):
    #     break
    #star_min, x_min, y_min, star_max, x_max, y_max = visualize_all_cells(i_star_cell,x_division_star_cell, y_division_star_cell,function_to_optimize2)
    #print(star_min <= x_min, star_min <= y_min)        
    #print(star_max >= x_max , star_max >= y_max)
    #if np.allclose(sub_optimality_after_x_division,sub_optimality_after_y_division) or sub_optimality_after_y_division == 0.0 or sub_optimality_after_x_division == 0.0:
    #    break
 
    if h < 2:
        division = 0
        i_star_cell = x_division_star_cell
     
    elif sub_optimality_after_x_division > sub_optimality_after_y_division:
        division = 0
        i_star_cell = x_division_star_cell
    else:
        division = 1
        i_star_cell = y_division_star_cell
    print(f"minimum after split along axis 1 is {sub_optimality_after_x_division} and {sub_optimality_after_y_division}")
    #assert i_star_cell_found == 2, "both i_star_cells not found, must me some issue"
    #divisions.append([h,division])
    # fig,axs = plt.subplots(1,1)
    # axs.scatter(i_star_cell[:,0],i_star_cell[:,1],c='r')
    # fig.savefig('temp.png')
    
    return i_star_cell, division, sub_optimality_after_x_division, sub_optimality_after_y_division


def choose_axis_to_split_function_variation_approach(i_star_cell, fun,maximum,h):
    axis1_variation,axis2_variation = function_variation_computation(i_star_cell,fun)

    print(f"axis1_variation: {axis1_variation}, axis2_variation: {axis2_variation}")
    # if np.allclose(axis1_variation,axis2_variation) or axis1_variation == 0 or axis2_variation == 0:
    #     return None, None
    i_star_cell_found = 0
    if axis1_variation > axis2_variation:
        division = 0
        #divisions.append([h,division])
        
        left_cell = i_star_cell.copy()
        left_cell[1] = i_star_cell[0] + (i_star_cell[1]-i_star_cell[0])/3.0
        left_cell[3] = i_star_cell[2] + (i_star_cell[3]-i_star_cell[2])/3.0
        i_star_h = point_in_parallelogram(left_cell[0],left_cell[1],left_cell[3],left_cell[2],maximum)
        if i_star_h:
            print(f"i star cell found at depth {h}")
            i_star_cell = left_cell.copy()
            i_star_cell_found +=1
        else:
            right_cell = i_star_cell.copy()
            right_cell[0] = i_star_cell[1] + (i_star_cell[0]-i_star_cell[1])/3.0
            right_cell[2] = i_star_cell[3] + (i_star_cell[2]-i_star_cell[3])/3.0
            i_star_h = point_in_parallelogram(right_cell[0],right_cell[1],right_cell[3],right_cell[2],maximum)
            if i_star_h:
                print(f"i star cell found at depth {h}")
                i_star_cell = right_cell.copy()
                i_star_cell_found +=1
            else:
                middle_cell = i_star_cell.copy()
                middle_cell[0] = left_cell[1]       
                middle_cell[1] = right_cell[0]
                middle_cell[2] = left_cell[3]        
                middle_cell[3] = right_cell[2]
                i_star_h = point_in_parallelogram(middle_cell[0],middle_cell[1],middle_cell[3],middle_cell[2],maximum)
                if i_star_h:
                    print(f"i star cell found at depth {h}")
                    i_star_cell = middle_cell.copy()
                    i_star_cell_found +=1
    else:
        division = 1
        #divisions.append([h,division])
        # suppose we decide to split along axis = 1
        bottom_cell = i_star_cell.copy()
        bottom_cell[3] = i_star_cell[1] + (i_star_cell[3] - i_star_cell[1])/3.0
        bottom_cell[2] = i_star_cell[0] + (i_star_cell[2] - i_star_cell[0])/3.0
        i_star_h = point_in_parallelogram(bottom_cell[0],bottom_cell[1],bottom_cell[3],bottom_cell[2],maximum)
        if i_star_h:
            print(f"i star cell found at depth {h}")
            i_star_cell = bottom_cell.copy()
            i_star_cell_found +=1
        else:

            top_cell = i_star_cell.copy()
            top_cell[0] = i_star_cell[2] + (i_star_cell[0] - i_star_cell[2])/3.0
            top_cell[1] = i_star_cell[3] + (i_star_cell[1] - i_star_cell[3])/3.0
            i_star_h = point_in_parallelogram(top_cell[0],top_cell[1],top_cell[3],top_cell[2],maximum)
            if i_star_h:
                print(f"i star cell found at depth {h}")
                i_star_cell = top_cell.copy()
                i_star_cell_found +=1
            else:
                middle_cell= i_star_cell.copy()
                middle_cell[0] = bottom_cell[2]
                middle_cell[1] = bottom_cell[3]
                middle_cell[2] = top_cell[0]
                middle_cell[3] = top_cell[1]
                i_star_h = point_in_parallelogram(middle_cell[0],middle_cell[1],middle_cell[3],middle_cell[2],maximum)
                if i_star_h:
                    print(f"i star cell found at depth {h}")
                    i_star_cell = middle_cell.copy()
                    i_star_cell_found +=1
    return i_star_cell, division