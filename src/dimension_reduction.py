
import numpy as np
import math
from src.utils.functions import print_status
from src.utils.rotations import labelled_corners, top_k_svd_directions
from src.utils.tie_break import top_k
from src.utils.nn import one_layer_net
import torch
import os, scipy
import yaml
from src.utils.rotations import sample_random_points,Y_scaler, dantzig_selector
from src.utils.general import set_seed, model_train, point_on_plane
from src.hyperparameter_training import neural_network_training
harmonic_number = lambda n: sum(1/i for i in range(1, n+1))
from src.utils.general import get_logger
import random

# # first able to reproduce the same results by fixing torch and numpy seeds
# rng1 = np.random.default_rng(123)   # numpy seed is fixed.
# # get the SeedSequence of the passed RNG
# ss = rng1.bit_generator._seed_seq
# # create 5 initial independent states
# num_of_trials = 1
# child_states = ss.spawn(num_of_trials)


# # torch seed needs to be fixed
# #seed = 123
# #torch.manual_seed(seed)
# #torch.cuda.manual_seed(seed)
# #torch.backends.cudnn.deterministic = True

torch.set_default_dtype(torch.float64)

# single random experiment repoducibility is working. Now I want to use joblib.
# I will use joblib to parallelize the experiments.




def stochastic_solver(function_instance, n, seed=0,torch_seed=0,**kwargs): 
    
    logger = get_logger(torch_seed, model_dir=kwargs.get('save_dir'))
    # setting the seeds
    rng = np.random.default_rng(seed)
    set_seed(torch_seed)
    
    # check is oracle_instance is a function class
    d = kwargs.get('d')
    m = kwargs.get('m')

    oracle = lambda x: -function_instance.evaluate_full(x)
    x_opt = function_instance.x_opt
    fmax = -function_instance.f_opt

    #int_opt = function_instance.int_opt
    cfg = kwargs.get('cfg',None)
    default_domain = np.vstack((function_instance.lb, function_instance.ub))
    logger.info(f"optimization domain is {default_domain}")


    # check if the x_opt is inside the domain.
    #assert np.all(x_opt >= function_instance.lb) and np.all(x_opt <= function_instance.ub) , 'x_opt is not in the domain of the function'

    # SequOOL parameters
    H_MAX = math.floor(n/harmonic_number(n))
    C = 1
    regret = []
    
    if cfg.dfo_method_args.dimension_reduction.use_true_subspace == True:
        directions = function_instance.r[:m].T
        finaly = float('-inf')
        finalx = None
        NUM_SAMPLES = 1

    if cfg.dfo_method_args.dimension_reduction.neural_network == True:
        # neural network training parameters
        X,Y = sample_random_points(cfg.dfo_method_args.all_dim_look_ahead.initial_samples,d,oracle,
                    rng=rng,bounds = default_domain)

        finaly = torch.max(Y).item()
        finalx = X[torch.argmax(Y)].numpy()

        best_model = neural_network_training(X,Y,cfg,logger,torch_seed,
                num_samples= cfg.dfo_method_args.all_dim_look_ahead.hyper_search_queries,gpus_per_trial=0)

        
        X = best_model.linear_one.weight.to('cpu').detach().numpy().T
        U_red, _, _, _ = top_k_svd_directions(X, reduction=True, M=m)
        directions = U_red
        angle = np.rad2deg(scipy.linalg.subspace_angles(directions, function_instance.r[:2].T))


    #-------------- end of training neural network-------------------
    
    
    #--------------SVD and finding the subspace--------------
        #X = model.linear_one.weight.to('cpu').detach().numpy().T * model.linear_two.weight.to('cpu').detach().numpy()
        #U_red, S_red, V_red, Orthonormal_basis = top_k_svd_directions(X,reduction=True,M=2)
        #directions = U_red      # U_red is a d \times p matrix
        
        #angle = np.rad2deg(scipy.linalg.subspace_angles(directions, function_instance.r[:2].T))
        
        # I want to retrain the model with different seed, till angle <=30
 
    if cfg.dfo_method_args.dimension_reduction.use_SIBO == True:
        logger.info(f"----------performing dantzig selection---------")
        # here I need to run the proecdure till I get flag = True from the function.
        M_PHI = cfg.dfo_method_args.dimension_reduction.SIBO.m_phi
        M_X = cfg.dfo_method_args.dimension_reduction.SIBO.m_x
        NUM_SAMPLES = M_PHI * M_X + M_X

        while True:
            V, function_values, flag, S = dantzig_selector(oracle, d, rng, logger, epsilon=1e-3, m_phi=M_PHI, m_x=M_X)
            if flag:
                break
            else:
                M_X = int(M_X*2)
                NUM_SAMPLES = M_PHI * M_X + M_X
               
        logger.info(f"number of samples used in dantzig selector are {NUM_SAMPLES}")
        directions = V[:m].T
        finaly = np.amax(np.array(function_values)).item()
        finalx = V[:1]    # TODO will change it later. This is not correct, but anyways we are not using it.
        regret.extend([fmax-finaly]*NUM_SAMPLES)
    #directions = function_instance.r[:2].T
    #directions = np.array([[1,0],[0,1],[0,0],[0,0]])
    M = directions.shape[1]   # subspace dimension dimension

    _, alpha_extent = labelled_corners(directions,default_domain)
    # TODO: add the scaling factor later.
    # making alpha_extent as square domain.
    #alpha_extent[0] = np.min(alpha_extent[0])
    #alpha_extent[1] = np.max(alpha_extent[1])
    
    alpha_extent = alpha_extent

    #eta = cfg.dfo_method_args.resoo.eta
        # [-m/eta, m/eta]^m
    #alpha_extent = np.array([[-cfg.low_dim/eta]*cfg.low_dim, [cfg.low_dim/eta]*cfg.low_dim])
    
    #------------- few assertions here----------
    try:
        angle = np.rad2deg(scipy.linalg.subspace_angles(directions, function_instance.r[:m].T))
        
        
        logger.info(f"subspace angle is {angle}")
        point_on_estimated_plane, x_star_on_plane, alpha = point_on_plane(function_instance.r[:m], directions.T, x_opt.T,alpha_extent)
            
        logger.info(f"point on the estimated plane is {point_on_estimated_plane}")
        logger.info(f"x_star_on_plane is {x_star_on_plane}")
    
        logger.info(f"alpha is {alpha}")
    except:
        logger.info("could not compute the angle")
        angle = float('inf')
        x_star_on_plane = False
        point_on_estimated_plane = None
        alpha = None
    #assert np.isclose(oracle(point_on_estimated_plane.T), fmax), "least squares estimation did not find point on plane"
    #assert x_star_on_plane == True, "x_star is not on the plane, need to increase the domain" 

    # if not x_star_on_plane:
    #     return None

        
    
    t = [{} for _ in range(H_MAX+2)]
    for i in range(H_MAX+2):
        t[i]['alpha_min'] = np.empty((0, M))
        t[i]['alpha_max'] = np.empty((0, M))
        t[i]['alpha_cen'] = np.empty((0, M))
        t[i]['cen_val'] = []

    t[0]['alpha_min'] = alpha_extent[0].reshape(1,-1)
    t[0]['alpha_max'] = alpha_extent[1].reshape(1,-1)
    
    
    t[0]['alpha_cen'] = ( ((t[0]['alpha_min'] + t[0]['alpha_max'])/2))
    t[0]['cen_val'] = [oracle( (directions @ t[0]['alpha_cen'].T ).T  )[0]]
    # initilisation of the tree
    
    sampled_value = t[0]['cen_val'][0]    # stores  the maximum value of a function 'f'
    
    #finaly = sampled_value
    #finalx = directions @ t[0]['alpha_cen'].T
    
    if sampled_value > finaly:
        finalx = directions @ t[0]['alpha_cen'].T
        finaly = sampled_value.item()
    n = NUM_SAMPLES; regret.append((fmax-finaly))
    
    count = 0
    for h in np.arange(0,H_MAX+1).reshape(-1):  # we cannot expand the bottom leaves.
                
        n_open_cells = 1 if h == 0 else min(int(C * math.floor(H_MAX/ h)), len(t[h]['alpha_cen']))
        #top_k_tuple = inf_tie_break(t[h]['cen_val'], t[h]['x'], n_open_cells)
        top_k_tuple = top_k(t[h]['cen_val'],n_open_cells)

        direction_to_split = count % directions.shape[1]
        count +=1
        for item in top_k_tuple:
            i_max = item[0]

            
            alpha_cen = t[h]['alpha_cen'][i_max]       # center in the alpha frame
            # alpha_cen_left stores the center of the left box
            alpha_cen_left = alpha_cen.copy()
            alpha_cen_left[direction_to_split] = (5 * t[h]['alpha_min'][i_max,direction_to_split] + t[h]['alpha_max'][i_max,direction_to_split]) / 6.0
            
            # alpha_cen_right stores the center of the right box
            alpha_cen_right = alpha_cen.copy()
            alpha_cen_right[direction_to_split] = (t[h]['alpha_min'][i_max,direction_to_split] + 5 * t[h]['alpha_max'][i_max,direction_to_split]) / 6.0
            
            
            # ------------------------left node -----------------------
            t[h + 1]['alpha_cen'] =np.concatenate((t[h+1]['alpha_cen'], alpha_cen_left.reshape(1, -1)), axis=0)
            
            # before sampling, we need to check if the point is inside the domain.
            point = (directions @ alpha_cen_left).T
            #point, flag = project_onto_boundary(directions, Orthonormal_basis, bounds, point)
            #if flag ==0:
            #    sampled_value = -np.inf
            #else:
                #sampled_value = oracle((directions @ alpha_cen_left).T).item()  # apply transformation
            sampled_value = oracle(point).item()
            n = n+1
            if sampled_value > finaly:
                finalx = directions @ alpha_cen_left
                finaly = sampled_value
            
                
            t[h + 1]['cen_val'].append(sampled_value)
            
            regret.append(max(fmax-finaly,0))
            print_status(d=d,n=n,h=h,i_max=i_max,x=directions @ alpha_cen_left,value=sampled_value,trees='t1')
            
            
            # splitting step # left child
            t[h + 1]['alpha_min'] =np.concatenate((t[h + 1]['alpha_min'], t[h]['alpha_min'][i_max].reshape(1,-1)), axis=0)
            
            newmax = t[h]['alpha_max'][i_max].copy()
            newmax[direction_to_split] = (2 * t[h]['alpha_min'][i_max,direction_to_split] + t[h]['alpha_max'][i_max,direction_to_split]) / 3.0
            
            t[h + 1]['alpha_max'] =np.concatenate((t[h + 1]['alpha_max'], newmax.reshape(1,-1)), axis=0)
            
            
            # ------------------------right node--------------------
            t[h + 1]['alpha_cen'] =np.concatenate((t[h+1]['alpha_cen'], alpha_cen_right.reshape(1,-1)), axis=0)
            sampled_value = oracle((directions @ alpha_cen_right).T).item()    
            
            if sampled_value > finaly:
                finalx = directions @ alpha_cen_right
                finaly = sampled_value
            n = n+1
            print_status(d=d,n=n,h=h,i_max=i_max,x=directions @ alpha_cen_right,value=sampled_value,trees='t1')
            t[h + 1]['cen_val'].append(sampled_value)
            
            regret.append(max(fmax-finaly,0))
            
            # splitting step # right child
            newmin = t[h]['alpha_min'][i_max].copy()
            newmin[direction_to_split] = (t[h]['alpha_min'][i_max,direction_to_split] + 2 * t[h]['alpha_max'][i_max,direction_to_split]) / 3.0

            t[h+1]['alpha_min'] = np.concatenate((t[h+1]['alpha_min'], newmin.reshape(1,-1)),axis=0)
            t[h+1]['alpha_max'] = np.concatenate((t[h+1]['alpha_max'],t[h]['alpha_max'][i_max].reshape(1,-1)), axis=0)
            
            
            #----------------------- central node ---------------------
            t[h+1]['alpha_cen'] = np.concatenate((t[h+1]['alpha_cen'],alpha_cen.reshape(1,-1)),axis=0)            
            t[h + 1]['cen_val'].append(t[h]['cen_val'][i_max])
            
            newmin = t[h]['alpha_min'][i_max,:].copy()
            newmax = t[h]['alpha_max'][i_max,:].copy()
            
            
            newmin[direction_to_split] = (2 * t[h]['alpha_min'][i_max,direction_to_split] + t[h]['alpha_max'][i_max,direction_to_split]) / 3.0
            newmax[direction_to_split] = (t[h]['alpha_min'][i_max,direction_to_split] + 2 * t[h]['alpha_max'][i_max,direction_to_split]) / 3.0

            t[h+1]['alpha_min'] = np.concatenate((t[h+1]['alpha_min'], newmin.reshape(1,-1)),axis=0)
            t[h+1]['alpha_max'] = np.concatenate((t[h+1]['alpha_max'], newmax.reshape(1,-1)),axis=0)
        
    #return finalx, finaly, regret,x_opt, fmax, x_star_on_plane, alpha, directions, angle, NUM_SAMPLES
    
    # reutrn it like a dictionary
    logger.info(f"regret is {regret[-1]}")
    logger.info(f"fmax is {fmax}")  
    return {
        'finalx': finalx,
        'finaly': finaly,
        'regret': regret,
        'x_opt': x_opt,
        'fmax': fmax,
        'x_star_on_plane': x_star_on_plane,
        'alpha': alpha,
        'directions': directions,
        'angle': angle,
        'NUM_SAMPLES': NUM_SAMPLES
    }
    
    # # Sample 100 uniformly distributed seeds from the range [0, 10^8]
    # torch_seeds = rng1.uniform(0, 1e8, size=num_of_trials).astype(int)
    # random_vector = Parallel(n_jobs=num_of_trials)(delayed(
    #     stochastic_function)(oracle_instance, n,random_state,torch_seed,**kwargs) for random_state,torch_seed in zip(child_states, torch_seeds))
    # return random_vector

    
    



