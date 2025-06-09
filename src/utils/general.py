from pathlib import Path
import os
import cvxpy as cp
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.optimize import Bounds
import scipy
import logging

def set_seed(seed: int = 42) -> None:
    # np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


# def optimizing_d(optimal_cells,rho):
#         n = len(optimal_cells)  # Number of linear constraints
#         # Define the optimization variable
#         x = cp.Variable(1)
#         # Define the objective function
#         objective = cp.Minimize(x[0])

#         # Define the constraints
#         constraints = []
#         for h in range(n):
#             constraint = -x[0] * optimal_cells[h][0] * np.log(rho) >= np.log(optimal_cells[h][1])
#             constraints.append(constraint)

#         # Additional constraint x[0] >= 0
#         constraints.append(x[0] >= 0)

#         # Solve the problem
#         problem = cp.Problem(objective, constraints)
#         problem.solve()

#         # Check if the problem was solved successfully
#         if problem.status == cp.OPTIMAL:
#             optimal_value = x.value[0]
#             print("Optimal value:", optimal_value)
#             sanity_check1 = [optimal_cells[h][1] <= rho**(-optimal_value*h) for h in range(len(optimal_cells))]
#             sanity_check2 = [optimal_cells[h][1] <= rho**(-0.99*optimal_value*h) for h in range(len(optimal_cells))]
#             return optimal_value, sanity_check1, sanity_check2
#         else:
#             print("Optimization problem did not converge.")
            
            
# def optimizing_d2(optimal_cells, rho,c=100):
#     # N(nu, rho, C) <= C*rho^(-d*h)
#     # Let C = 1
#     # d_h = -log(N(nu, rho, C))/log(rho)/h
#     # d = max {d_h}
#     # d = max {-log(N(nu, rho, C))/log(rho)/h}    
#     #d1 = np.max([-np.log(cells[1])/(np.log(rho)*cells[0]) for cells in optimal_cells if cells[0]>=1])
#     d1 = np.max([(np.log(2)-np.log(cell[1]))/(np.log(rho)*cell[0]) for cell in optimal_cells if cell[0]>0])
#     #d2 = np.max([-np.log(cells[1])/(np.log(rho)*cells[0]) for cells in optimal_cells if cells[0]>=5])
#     d2 = np.max([(np.log(10)-np.log(cell[1]))/(np.log(rho)*cell[0]) for cell in optimal_cells if cell[0]>0])
#     dc = np.max([(np.log(c)-np.log(cell[1]))/(np.log(rho)*cell[0]) for cell in optimal_cells if cell[0]>0])
#     return d1,d2,dc

# def rotated_vectors(angle,d=2):
#     directions_use = np.zeros_like(np.eye(d))
#     new_v1 = np.array([np.cos(angle),np.sin(angle)])
#     new_v2 = np.array([-np.sin(angle),np.cos(angle)])
#     directions_use[:, 0] = new_v1
#     directions_use[:, 1] = new_v2
#     return directions_use

def get_grid_points(x_min, x_max, num_points=10):
    """
    Args:
        x_min (np.ndarray): Minimum coordinates of the hyperrectangle.
        x_max (np.ndarray): Maximum coordinates of the hyperrectangle.
        
    Returns:
        np.ndarray: Array of all points within the hyperrectangle.
    """
    dimensions = len(x_min)
    
    # Create a list of ranges for each dimension
    ranges = [np.linspace(x_min[i], x_max[i] , num_points) for i in range(dimensions)]

    # Use np.meshgrid to create the grid of points
    grid = np.meshgrid(*ranges,indexing='ij')
    grid = np.stack(grid,axis=0)
    # Flatten the grid and return the result
    return grid, np.column_stack([g.flat for g in grid])


def cell_function_minimum(A,alpha_min, alpha_max, f_hat_numpy,rng,num_points=100):
    
    # this can be done more efficiently, since we are working with alpha domain now.
    bnds = Bounds(alpha_min, alpha_max)
    def objective(alpha):
       return f_hat_numpy(A @ alpha)
    #res = minimize(objective, x0= (alpha_max+ alpha_min)/2, method='SLSQP', bounds=bnds)
    
    #rng = np.random.default_rng(123)
    #samples = rng.uniform(
    #   low=alpha_min,
    #   high=alpha_max,
    #   size = (num_points, A.shape[0])
    #)
    #mean = np.mean(np.sort(f_hat_numpy((A @ samples.T).T).ravel())[:10])
    # # differential evolution
    res = scipy.optimize.differential_evolution(objective,x0= (alpha_max+ alpha_min)/2,
                             bounds = bnds )

    assert np.all(alpha_min <= res.x) and np.all(res.x <= alpha_max), 'optimization did not found a point inside the domain'
    
    # grid,_ = get_grid_points(alpha_min, alpha_max,num_points=num_points)
    # M = A.shape[1]
    
    # points_in_original_frame = np.matmul(A, grid.reshape(M,-1)) #.reshape(grid.shape)
    # #points_in_original_frame = np.moveaxis(points_in_original_frame, 0, -1)  # of shape ([50]*d, d)
    # #function_values = f_hat_numpy(points_in_original_frame)   # of shape ([50]*d, 1)
    # function_values = f_hat_numpy(points_in_original_frame.T)
    # function_values = np.squeeze(function_values)   # of shape ([50]*d)

    return res.fun #np.amin(function_values)

def model_train(model,optimizer,X,Y,epochs,scheduler):
    
    print(f"Retraining the model with {X.shape[0]} samples")
    print(f"loss before retraining: {model.criterion(model(X),Y)}")
    model.fit(X,Y,epochs,optimizer,scheduler)
    print(f"loss after retraining: {model.criterion(model(X),Y)}")
    
    # fig, axs = plt.subplots(1,1)
    # axs.scatter(x_samples[:,0],x_samples[:,1],c=y_samples)
    # fig.savefig('retraining_samples.png')
    return model.criterion(model(X),Y)
    
def model_train_dataloader(model,optimizer,dataloader,epochs,scheduler):
    
    print(f"Retraining the model with {X.shape[0]} samples")
    print(f"loss before retraining: {model.criterion(model(X),Y)}")
    model.fit(dataloader,epochs,optimizer,scheduler)
    print(f"loss after retraining: {model.criterion(model(X),Y)}")
    
    # fig, axs = plt.subplots(1,1)
    # axs.scatter(x_samples[:,0],x_samples[:,1],c=y_samples)
    # fig.savefig('retraining_samples.png')
    
def point_on_plane(A, A_hat, x_star,alpha_extent):
    alpha = np.linalg.lstsq(A.T @ A @ A_hat.T, A.T @ A @ x_star,rcond=None)[0]
    #alpha = np.linalg.inv(A_hat @ A_hat.T) @ A @ x_star
    point_on_estimated_plane = A_hat.T @ alpha
    
    x_star_on_plane = np.all(np.logical_and(alpha_extent[0] <= np.squeeze(alpha), np.squeeze(alpha) <= alpha_extent[1]))
    return point_on_estimated_plane, x_star_on_plane, alpha


import numpy as np
import pickle
# from matplotlib import pyplot as plt
def joblib_pickle_load(file_name, method=None, **kwargs):
    ### Dimension Reduction look ahead ----
    with open(file_name, 'rb') as f:
        data = pickle.load(f) # deserialize using load()


    if method in ['direct', 'dual_annealing']:
        finaly = np.array([data[i][0].fun for i in range(len(data))])
        evaluations = np.array([data[i][0].nfev for i in range(len(data))])
        fmin = np.array([data[i][1] for i in range(len(data))])
        regret = finaly-fmin
        x_star_array = np.array([data[i][2] for i in range(len(data))])
    
        return regret, x_star_array, fmin, evaluations
        
    if method in ['dim_red', 'dim_red_look']:
        booleans = [data[i][6] for i in range(len(data))]
        regrets = [data[i][3] for i in range(len(data)) if booleans[i]]
        fmax =  np.array([data[i][5] for i in range(len(data)) ])
        x_star_array = np.array([data[i][4] for i in range(len(data))])
        return np.array(regrets), x_star_array, fmax,booleans, data
    
    if method in ['dim_red_cs','dim_red_nn','dim_red_cs_lookahead']:
        
        try: 
            booleans = [data[i][6] for i in range(len(data))]
            regrets = [data[i][3] for i in range(len(data)) if booleans[i]]
        # preprocessing for variable length regrets
        
        #max_length_regret = max([len(regrets[i]) for i in range(len(regrets))])
        #[regrets[i].extend([regrets[i][-1]]*(max_length_regret - len(regrets[i]))) for i in range(len(regrets))]
        
        #min_length_regret = min([len(regrets[i]) for i in range(len(regrets))])
        #regrets = [regrets[i][:min_length_regret] for i in range(len(regrets))]
        
            fmax =  np.array([data[i][5] for i in range(len(data)) ])
            x_star_array = np.array([data[i][4] for i in range(len(data))])            
        except:
            booleans = [data[i][5] for i in range(len(data))]
            regrets = [data[i][2] for i in range(len(data)) if booleans[i]]
            fmax =  np.array([data[i][4] for i in range(len(data)) ])
            x_star_array = np.array([data[i][3] for i in range(len(data))])
            
        return np.array(regrets), x_star_array, fmax,booleans, data
        
    if method == 'sequool':
        try:
            regrets = np.array([data[i][3] for i in range(len(data))])
            x_star_array = np.array([data[i][4] for i in range(len(data))])
            fmax = np.array([data[i][5] for i in range(len(data))])
        except:
            regrets = np.array([data[i][2] for i in range(len(data))])
            x_star_array = np.array([data[i][3] for i in range(len(data))])
            fmax = np.array([data[i][4] for i in range(len(data))])
        
        return regrets, x_star_array, fmax,data
    
    if method == 'random_search':
        finaly = np.array([data[i][0] for i in range(len(data))])
        fmin = np.array([data[i][1] for i in range(len(data))])
        regret = finaly-fmin
        x_star_array = np.array([data[i][2] for i in range(len(data))])
        return regret, x_star_array, fmin, kwargs.get('n')

def pickle_plot_data(axs, data, label, **kwargs):
    offset = kwargs.get('offset',0)
    mean_type = kwargs.get('mean_type', 'mean')
    alpha = kwargs.get('alpha', 0.2)
    if mean_type == 'median':
        mean = np.median(data, axis=0)
        mean = np.percentile(data, 50, axis=0)
        quantile_95 = np.percentile(data, 85, axis=0)
        quantile_0 = np.percentile(data, 0, axis=0)
        std = np.median(np.absolute(data - np.median(data,axis=0)),axis=0) # np.std(data, axis=0)
        x = np.arange(offset,data.shape[1]+offset)
        axs.plot(x, mean, label = label)
        axs.fill_between(x, quantile_0, quantile_95, alpha=alpha)
    else:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) #
        std_err = std / np.sqrt(data.shape[0])
        x = np.arange(offset,data.shape[1]+offset)
        axs.plot(x, mean, label = label)
        axs.fill_between(x, mean - 1.5*std_err, mean + 1.5*std_err, alpha=0.2) 
def pickle_plot_data2(axs,data,label,marker):
    # this is for direct, annealing, random_search
    mean = np.mean(data[0])
    
    std_err = np.std(data[0]) / np.sqrt(data[0].shape[0])
    print(std_err)
    average_evaluations = np.mean(data[3]).item()
    #axs.scatter([average_evaluations], [mean], label = label,marker = marker)
    axs.errorbar(average_evaluations, mean, yerr=1.5*std_err,alpha = 1,label = label,marker = marker,fmt='o',ecolor='g')
    #print(average_evaluations, mean-1.5*std_err)
    #axs.fill(average_evaluations, mean - 1.5*std_err, mean + 1.5*std_err, alpha=0.2)  
    #axs.axvspan(average_evaluations-1, average_evaluations+1, ymin=mean - 1.5*std_err, ymax=mean + 1.5*std_err, alpha=1, color='gray')
    
def validate_data(seq_data, dim_red_data, direct_data, dual_annealing_data, random_search_data):
    assert np.array_equal(seq_data[2], dim_red_data[2])
    assert np.array_equal(-direct_data[2], dim_red_data[2])
    assert np.array_equal(dual_annealing_data[2], direct_data[2])
    assert np.array_equal(random_search_data[2], dual_annealing_data[2])

    assert np.array_equal(seq_data[1], dim_red_data[1])
    assert np.array_equal(direct_data[1], dim_red_data[1])
    assert np.array_equal(dual_annealing_data[1], dim_red_data[1])
    assert np.array_equal(random_search_data[1], dual_annealing_data[1])
   
def load_data(*paths):
    data = {}
    for path in paths:
        name, file_path = path.split(',')
        if name == 'random_search':
            data[name] = joblib_pickle_load(file_path, name,n=10000)
        else:
            data[name] = joblib_pickle_load(file_path, name)
    return data 


# def load_data(seq_path, dim_red_path, dim_red_path_cs, dim_red_path_cs_lookahead, direct_path, dual_annealing_path, random_search_path):
    
#     seq_data = joblib_pickle_load(seq_path, 'sequool')
#     dim_red_data = joblib_pickle_load(dim_red_path, 'dim_red')
#     dim_red_data_cs = joblib_pickle_load(dim_red_path_cs, 'dim_red_cs')
#     dim_red_data_cs_lookahead = joblib_pickle_load(dim_red_path_cs_lookahead, 'dim_red_cs_lookahead')
#     direct_data = joblib_pickle_load(direct_path, 'direct')
#     dual_annealing_data = joblib_pickle_load(dual_annealing_path, 'dual_annealing')
#     random_search_data = joblib_pickle_load(random_search_path, 'random_search', n=10000)
    

#     return seq_data, dim_red_data, dim_red_data_cs, direct_data, dual_annealing_data, random_search_data, dim_red_data_cs_lookahead

# def set_logger(model_dir, log_name):
#     '''Set logger to write info to terminal and save in a file.

#     Args:
#         model_dir: (string) path to store the log file

#     Returns:
#         None
#     '''
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)

#     #Don't create redundant handlers everytime set_logger is called
#     if not logger.handlers:

#         #File handler with debug level stored in model_dir/generation.log
#         fh = logging.FileHandler(os.path.join(model_dir, log_name))
#         fh.setLevel(logging.DEBUG)
#         fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
#         logger.addHandler(fh)

#         #Stream handler with info level written to terminal
#         sh = logging.StreamHandler()
#         sh.setLevel(logging.INFO)
#         sh.setFormatter(logging.Formatter('%(message)s'))
#         logger.addHandler(sh)
    
#     return logger

def get_logger(worker_id, model_dir):
    logger_name = f'worker_{worker_id}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create a log file for this worker process in the specified model_dir
    log_file = os.path.join(model_dir, f'{logger_name}.log')
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    # Add a stream handler to print logs to the terminal
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    
    return logger