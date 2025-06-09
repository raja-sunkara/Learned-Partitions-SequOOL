
import numpy as np
import math
from src.utils.rotations import labelled_corners, top_k_svd_directions
from src.utils.tie_break import top_k
import torch
from src.utils.rotations import sample_random_points, X_scaler_high_d, Y_scaler, dantzig_selector
from src.utils.general import set_seed
import itertools
from src.utils.functions import print_status
from src.utils.general import get_grid_points, cell_function_minimum, model_train

from src.hyperparameter_training import neural_network_training
from src.utils.general import get_logger


harmonic_number = lambda n: sum(1/i for i in range(1, n+1))


def stochastic_function(function_instance, n, seed, torch_seed, **kwargs):
    
    logger = get_logger(torch_seed, model_dir=kwargs.get('save_dir'))
    rng = np.random.default_rng(seed)
    torch.set_default_dtype(torch.float64)
    set_seed(torch_seed)
    TORCH_DTYPE = torch.float64


    regret = []
    d = kwargs.get('d')

    fmax = -function_instance.f_opt
    oracle = lambda x: -function_instance.evaluate_full(x)
    x_opt = function_instance.x_opt
    
    cfg = kwargs.get('cfg', None)
    default_domain = np.vstack((function_instance.lb, function_instance.ub))
    bounds  = default_domain
    logger.info(f"optimization domain is {default_domain}")

    
    # neural network training
    X,Y = sample_random_points(cfg.dfo_method_args.all_dim_look_ahead.initial_samples,d,oracle,
                    rng=rng,bounds = bounds)

    #best_model = neural_network_training(X,Y,cfg,logger,torch_seed,num_samples= cfg.dfo_method_args.all_dim_look_ahead.hyper_search_queries
    #, gpus_per_trial=0)

    #f_hat_numpy = lambda x:  best_model(torch.tensor(x).type(TORCH_DTYPE)).detach().numpy()


    C = 1 #for (3^3,3)
    
            
    finaly = torch.max(Y).item()
    finalx = X[torch.argmax(Y)].numpy()
    #regret.extend([fmax-finaly]*X.shape[0])
    
   
    #visualize_nn_model(model, f, 0, 1, 0), # Visualization is not possible in high-d   
    #-------------------------- end of neural network training------------
    
    
    #-------------------- SVD and finding the subspace----------
    logger.info(f"----------performing dantzig selection---------")
    M_PHI = cfg.dfo_method_args.dimension_reduction.SIBO.m_phi
    M_X = cfg.dfo_method_args.dimension_reduction.SIBO.m_x
    NUM_SAMPLES = M_PHI * M_X + M_X

    V, function_values, flag,S = dantzig_selector(oracle, d, rng, 
                                    logger, epsilon=1e-3, m_phi=M_PHI, m_x=M_X)

    logger.info(f"number of samples used in dantzig selector are {NUM_SAMPLES}")
    directions = V[:d].T
    #X = model.linear_one.weight.to('cpu').detach().numpy().T * model.linear_two.weight.to('cpu').detach().numpy()
    #U_red, S_red, V_red, _ = top_k_svd_directions(X,reduction=False)
    #Need to add a check here to select the number of directions. Or use, all the d directions and let look a head decide which axis to split.
    #directions = U_red      # U_red is a d \times p matrix
    directions = np.eye(d)
    #directions = np.eye(d)  # for now, use all the directions
    #directions = oracle_instance.r.T
    _, alpha_extent = labelled_corners(directions,bounds)

    alpha_star = x_opt @ directions
    if not np.all(alpha_extent[0] <= alpha_star) and np.all(alpha_star <= alpha_extent[1]):
        return None

    #grouped_points = group_labelled_corners(labeled_points,directions.shape[1])
    
    #tree = grouped_points[np.newaxis]
    # grouped points is of shape (d, 2^(d-1),2,d)
#    h_max = n/log_bar(n)
    H_MAX = math.floor(n/harmonic_number(n))

    # different line of tought, sample more in the first axis and less in the last axis
    # k = H_MAX/harmonic_number(d)
    # axis_sampling_distribution = np.array([k/i for i in range(1,d+1)])
    # axis_sampling_distribution = axis_sampling_distribution/sum(axis_sampling_distribution)

    # axis_sampling_distribution = S / np.sum(S)


    # samples_axis = np.random.default_rng(123).choice(len(axis_sampling_distribution), size=H_MAX+1, p=axis_sampling_distribution)
    round_robin = False
    counter = 0
    t = [{} for _ in range(H_MAX+2)]
    for i in range(H_MAX+2):
        t[i]['alpha_min'] = np.empty((0, d), dtype=float)
        t[i]['alpha_max'] = np.empty((0, d), dtype=float)
        t[i]['alpha_cen'] = np.empty((0, d), dtype=float)
        t[i]['cen_val'] = []

    t[0]['alpha_min'] = alpha_extent[0].reshape(1,-1)
    t[0]['alpha_max'] = alpha_extent[1].reshape(1,-1)
    
    

    t[0]['alpha_cen'] = ( ((t[0]['alpha_min'] + t[0]['alpha_max'])/2))
    t[0]['cen_val'] = [oracle(  (directions @ t[0]['alpha_cen'].T ).T  )[0]]
    # ## initilisation of the tree
    
    sampled_value = t[0]['cen_val'][0]
    if sampled_value > finaly:
        finaly = sampled_value
        finalx = directions @ t[0]['alpha_cen'].T
    
    #finaly, finalx = t[0]['cen_val'][0], directions @ t[0]['alpha_cen'].T    # stores  the maximum value of a function 'f'
    n = 1; regret.append((fmax-finaly))
    
    
    # need to find $i^*$ cell, and find the sequnce of axis to split.
    # initally $i^$ cell is the grouped points
    #i_star_cell = grouped_points[0].reshape(-1,d)
    
    # store dictionary to count how many times each axis is split
    axis_divisions = {i:0 for i in range(d)}
    for h in np.arange(0,H_MAX+1).reshape(-1):  # we cannot expand the bottom leaves.
        

        # --------- This Procedure is to find the axis to split at each depth. ------------
        # at each depth, looking at the i_star_cell and finding the axis to split.
        # Based on the Zoom discussion on 22-Apr. 
        # i^{star} cell is the largest function value cell.            
        n_open_cells = 1 if h == 0 else min(int(C * math.floor(H_MAX/ h)), len(t[h]['alpha_cen']))

        
        #top_k_tuple = inf_tie_break(t[h]['cen_val'], t[h]['x'], n_open_cells)
        top_k_tuple = top_k(t[h]['cen_val'],n_open_cells)


        #-------------------------- NN Look ahead approach ---------------------
        if not round_robin:
            # i-star-cell is the top function representative cell
            i_star_cell_index = top_k_tuple[0][0]
            # Now, Identify the axis to split using look ahead strategy
                    
            minimum_values = []
            
            for axis in range(d):
                # middle cell is the i^star cell
                newmin = t[h]['alpha_min'][i_star_cell_index,:].copy()   # copy the i^{star} cell
                newmax = t[h]['alpha_max'][i_star_cell_index,:].copy()   # copy the i^{star} cell

                # newmin, newmax descirbe the i^{star} cell after dividing along axis
                newmin[axis] = (2 * t[h]['alpha_min'][i_star_cell_index,axis] + t[h]['alpha_max'][i_star_cell_index,axis]) / 3.0
                newmax[axis] = (t[h]['alpha_min'][i_star_cell_index,axis] + 2 * t[h]['alpha_max'][i_star_cell_index,axis]) / 3.0
                
                function_minimum = cell_function_minimum(directions, newmin, newmax, oracle,rng,num_points=int(cfg.dfo_method_args.all_dim_look_ahead.num_point_for_minimum))
                minimum_values.append(function_minimum)
            direction_to_split = np.argmax(np.array(minimum_values))
            #direction_to_split = samples_axis[h]
            axis_divisions[direction_to_split] += 1
        else:
            direction_to_split = counter % d
            counter += 1
            axis_divisions[direction_to_split] += 1
        
        #---------------------------end of NN Look ahead--------------------------
        
        for item in top_k_tuple:
            i_max = item[0]

            

            # now we have the direction to split, we can split the cell along this direction 
            xx = t[h]['alpha_cen'][i_max]       # center in the alpha frame
            # x_g stores the center of the left box
            x_g = xx.copy()
            x_g[direction_to_split] = (5 * t[h]['alpha_min'][i_max,direction_to_split] + t[h]['alpha_max'][i_max,direction_to_split]) / 6.0
            
            # x_d stores the center of the right box
            x_d = xx.copy()
            x_d[direction_to_split] = (t[h]['alpha_min'][i_max,direction_to_split] + 5 * t[h]['alpha_max'][i_max,direction_to_split]) / 6.0
            
            
            # ------------------------left node -----------------------
            t[h + 1]['alpha_cen'] =np.concatenate((t[h+1]['alpha_cen'], x_g.reshape(1, -1)), axis=0)
            # here, we need a check,
            
            query_point = directions @ x_g
            if True: #np.all(np.logical_and(bounds[0] <= query_point, query_point <= bounds[1])):
                sampled_value = oracle((query_point).T).item()  # apply transformation
                n = n+1
                if sampled_value > finaly:
                    finalx = directions @ x_g
                    finaly = sampled_value
                regret.append(fmax-finaly)
            else:
                sampled_value = -np.inf
                                    
            t[h + 1]['cen_val'].append(sampled_value)
            
            
            print_status(d=d,n=n,h=h,i_max=i_max,x=directions @ x_g,value=sampled_value,trees='t1')
            
            
            # splitting step # left child
            t[h + 1]['alpha_min'] =np.concatenate((t[h + 1]['alpha_min'], t[h]['alpha_min'][i_max].reshape(1,-1)), axis=0)
            
            newmax = t[h]['alpha_max'][i_max].copy()
            newmax[direction_to_split] = (2 * t[h]['alpha_min'][i_max,direction_to_split] + t[h]['alpha_max'][i_max,direction_to_split]) / 3.0
            
            t[h + 1]['alpha_max'] =np.concatenate((t[h + 1]['alpha_max'], newmax.reshape(1,-1)), axis=0)
            
            
            # ------------------------right node--------------------
            t[h + 1]['alpha_cen'] =np.concatenate((t[h+1]['alpha_cen'], x_d.reshape(1,-1)), axis=0)
            query_point = directions @ x_d
            if True: #np.all(np.logical_and(bounds[0] <= query_point, query_point <= bounds[1])):
                sampled_value = oracle((query_point).T).item()  # apply transformation
                n = n+1
                if sampled_value > finaly:
                    finalx = directions @ x_d
                    finaly = sampled_value
                regret.append(fmax-finaly)
            else:
                sampled_value = -np.inf
                            
            print_status(d=d,n=n,h=h,i_max=i_max,x=directions @ x_d,value=sampled_value,trees='t1')
            t[h + 1]['cen_val'].append(sampled_value)
            
            
            
            # splitting step # right child
            newmin = t[h]['alpha_min'][i_max].copy()
            newmin[direction_to_split] = (t[h]['alpha_min'][i_max,direction_to_split] + 2 * t[h]['alpha_max'][i_max,direction_to_split]) / 3.0

            t[h+1]['alpha_min'] = np.concatenate((t[h+1]['alpha_min'], newmin.reshape(1,-1)),axis=0)
            t[h+1]['alpha_max'] = np.concatenate((t[h+1]['alpha_max'],t[h]['alpha_max'][i_max].reshape(1,-1)), axis=0)
            
            
            #----------------------- central node ---------------------
            t[h+1]['alpha_cen'] = np.concatenate((t[h+1]['alpha_cen'],xx.reshape(1,-1)),axis=0)            
            t[h + 1]['cen_val'].append(t[h]['cen_val'][i_max])
            
            newmin = t[h]['alpha_min'][i_max,:].copy()
            newmax = t[h]['alpha_max'][i_max,:].copy()
            
            
            newmin[direction_to_split] = (2 * t[h]['alpha_min'][i_max,direction_to_split] + t[h]['alpha_max'][i_max,direction_to_split]) / 3.0
            newmax[direction_to_split] = (t[h]['alpha_min'][i_max,direction_to_split] + 2 * t[h]['alpha_max'][i_max,direction_to_split]) / 3.0

            t[h+1]['alpha_min'] = np.concatenate((t[h+1]['alpha_min'], newmin.reshape(1,-1)),axis=0)
            t[h+1]['alpha_max'] = np.concatenate((t[h+1]['alpha_max'], newmax.reshape(1,-1)),axis=0)

        if False: #(h+1) >= cfg.dfo_method_args.all_dim_look_ahead.train_till_height:
            round_robin = True  
        
        if False:#not round_robin:
            if (h+1)% cfg.dfo_method_args.all_dim_look_ahead.neural_network_training_freq ==0:
                
                    # need to retrain neural network by collecting the samples on the smaller domain
                    #model = one_layer_net(d, n_hidden, 1)   # instantiate new nn model
                    #optimizer = torch.optim.Adam(model.parameters(),weight_decay=0.5, lr=0.0001)

                # get the i^* cell
                top_k_tuple = top_k(t[h+1]['cen_val'],n_open_cells)
                i_star_cell_index = top_k_tuple[0][0]
                newmin = t[h+1]['alpha_min'][i_star_cell_index,:].copy()
                newmax = t[h+1]['alpha_max'][i_star_cell_index,:].copy()
                # these are the points in alpha space, construct 2^d corners in real space
                #extent = np.vstack((newmin, newmax))
                #two_power_d = list(itertools.product(*extent.T))
                #two_power_d_corners = (directions @ np.array(two_power_d).T).T
            
                
                X,Y = sample_random_points(cfg.dfo_method_args.all_dim_look_ahead.initial_samples,d,oracle,
                    rng=rng,bounds = [newmin, newmax])
                
#                X,Y = sample_random_points(RETRAINING_NUM_SAMPLES,d,oracle,rng=rng, cell=two_power_d_corners)

                # add the same regret value for the RETRAINING_NUM_SAMPLES times
                regret.extend([fmax-finaly]*X.shape[0])
                #here, need to scale the data
                # x_scaler = X_scaler_high_d(two_power_d_corners)
                # y_scaler = Y_scaler()
                
                # x_transformed = (x_scaler.fit_transform(X)).type(TORCH_DTYPE)
                # y_transformed = y_scaler.fit_transform(Y)

                # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
                # #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,momentum=MOMENTUM)
                # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

                # model_train(model,optimizer, x_transformed,y_transformed,EPOCHS,scheduler)
                # # here, we trained model on the scaled data.

                # since, model is trained on the scaled data, we need to scale the data before sending to the model.
                #del f_hat_numpy

                #best_model = neural_network_training(X,Y,cfg,logger,torch_seed,num_samples= cfg.dfo_method_args.all_dim_look_ahead.hyper_search_queries
                #    , gpus_per_trial=0)

                #f_hat_numpy = lambda x:\
                #best_model(torch.tensor((x.reshape(-1,d))).type(TORCH_DTYPE)).detach().numpy()
                

                average_fit_error = torch.mean((Y - torch.mean(Y))**2)
                
                if best_model.criterion(best_model(X),Y) < average_fit_error:
                    print(f"Model fit is better than average value fit")
                else:
                    print(f"Model fit is not better than average value fit")
                    round_robin = True
            

    # return finalx,finaly,t,regret

    return {
        'finalx': finalx,
        'finaly': finaly,
        'regret': regret,
        'axis_divisions': axis_divisions,
    }



    



