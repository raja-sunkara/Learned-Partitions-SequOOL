
import numpy as np
import math
from src.utils.functions import print_status
from src.utils.rotations import labelled_corners, top_k_svd_directions
from src.utils.tie_break import top_k
from src.utils.nn import one_layer_net
import torch
from src.utils.rotations import sample_random_points
from src.utils.rotations import X_scaler_high_d, Y_scaler, dantzig_selector
import itertools
from src.utils.general import set_seed, model_train, point_on_plane, cell_function_minimum
import os, sys, yaml, scipy
from matplotlib import pyplot as plt
from src.utils.general import get_logger
harmonic_number = lambda n: sum(1/i for i in range(1, n+1))

# # first able to reproduce the same results by fixing torch and numpy seeds
# rng1 = np.random.default_rng(123)   # numpy seed is fixed.
# # get the SeedSequence of the passed RNG
# ss = rng1.bit_generator._seed_seq
# # create 5 initial independent states
# num_of_trials = 1
# child_states = ss.spawn(num_of_trials)

torch.set_default_dtype(torch.float64)


def visualize_nn_model(f_hat_numpy,true_f,x_min,x_max,h,x_scaler,y_scaler):
    
    fig, axs = plt.subplots(1,2)
    x = np.linspace(x_min,x_max,100)
    y = np.linspace(x_min,x_max,100)
    xx,yy = np.meshgrid(x,y)
    
    data_x = np.stack((xx,yy),axis=-1).reshape(-1,2)
    z = true_f(data_x)
    z_nn = f_hat_numpy(data_x)
    fig, axs = plt.subplots(1,2)
    img1 = axs[0].scatter(data_x[:,0],data_x[:,1],c = z)
    #img1 = axs[0].imshow(z.reshape((xx.shape[0],xx.shape[0])), origin='lower', extent=(x_min,x_max,x_min,x_max),cmap='plasma')
    plt.colorbar(img1, ax=axs[0])

    img2 = axs[1].scatter(data_x[:,0],data_x[:,1],c = z_nn)

    plt.colorbar(img2,ax=axs[1])
    # save it inside save_dir
    save_path = os.path.join(f'nn_model_{h+1}.png')
    fig.savefig(save_path)
    plt.close(fig)

     

    
def stochastic_solver(oracle_instance, budget, seed, torch_seed, **kwargs):
    
    logger = get_logger(torch_seed, model_dir=kwargs.get('save_dir'))
    # numpy RNG child
    #seed = 0
    #torch_seed = 0
    rng = np.random.default_rng(seed)
    set_seed(torch_seed)
    
    # check is oracle_instance is a function class
    d = kwargs.get('d')
    m = kwargs.get('m')
    int_opt = kwargs.get('int_opt')
    function_instance = oracle_instance(d, rng=rng, int_opt = int_opt, sub_space_dim = m)
    fmax = -function_instance.f_opt
    
    # try:
    #     x_opt = function_instance.x_opt[:,1].T
    # except:
    x_opt = function_instance.x_opt
        
    oracle = lambda x: -function_instance.evaluate_full(x)
    
    
    
    # # function parameters
    # d = oracle_instance.d
    # fmax = -oracle_instance.f_opt
    # oracle = lambda x: -oracle_instance.evaluate_full(x)
    # [-5,5]^d is the default domain
    default_domain = np.array([[-5, 5]]*d).T
    
    bounds = kwargs.get('bounds', default_domain)
    logger.info(f"optimization domain is {bounds}")
            
    # Neural network training parameters
    function_name = function_instance.__class__.__name__
    home_directory = os.path.expanduser('~')
    yaml_file_name=  home_directory + f'/research/bb_optimization/nn_hyper/{function_name.lower()}.yaml'# os.path.join(root_directory, )
    
    # Load the configuration file
    with open(yaml_file_name, 'r') as f:
        config = yaml.safe_load(f)
    
        #Neural network constants        
    NUM_SAMPLES =config['NUM_SAMPLES'] + 300
    RETRAINING_NUM_SAMPLES = 100
    NUM_HIDDEN = config['NUM_HIDDEN'] 
    
    EPOCHS = config['EPOCHS']
    LEARNING_RATE = config['LEARNING_RATE']
    WEIGHT_DECAY = config['WEIGHT_DECAY']
    MOMENTUM = config['MOMENTUM']
    GAMMA = config['GAMMA']
    STEP_SIZE = config['STEP_SIZE']
    TORCH_DTYPE = torch.float64
    
    logger.info(f"neural network configuration file{config}")
    # SequOOL parameters
    H_MAX = math.floor(budget/harmonic_number(budget))
    C =1

    regret = []
    #regret.append(fmax)    
    # --------------training the neural network at depth=0-----------------
    model = one_layer_net(d, NUM_HIDDEN, 1) 
    # model.to('cuda') 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    X,Y = sample_random_points(NUM_SAMPLES,d,oracle, rng=rng,bounds = bounds)
    #finaly = torch.max(Y).item()
    #finalx = X[torch.argmax(Y)].numpy()
    
    #regret.extend([fmax-finaly]*NUM_SAMPLES)
    
    y_scaler = Y_scaler()
    y_transformed = y_scaler.fit_transform(Y)
    
    # X = X.to('cuda')
    # y_transformed = y_transformed.to('cuda')
    logger.info(f"neural network training has started")
    loss = model_train(model,optimizer,X,y_transformed,EPOCHS,scheduler)
    logger.info(f"loss after the neural network training is {loss}")
    # model.to('cpu') 
    #-------------- end of training neural network-------------------
    
    f_hat_numpy = lambda x: y_scaler.inverse_transform(model(torch.tensor(x).type(TORCH_DTYPE)).detach().numpy())
    # lets visualize the neural network. 
    if d==2:
        visualize_nn_model(f_hat_numpy,oracle, x_min=-5, x_max=5, h=0, x_scaler =False, y_scaler = False)
        
            
    #--------------SVD and finding the subspace--------------
    # for subspace, will use the dantzig selector
    neural_network = False
    
    
    if neural_network:
        X = model.linear_one.weight.to('cpu').detach().numpy().T * model.linear_two.weight.to('cpu').detach().numpy()
        U_red, S_red, V_red, Orthonormal_basis = top_k_svd_directions(X,reduction=True,variance=0.85,M=2)
        directions = U_red      # U_red is a d \times p matrix
    else:
        # here I need to run the proecdure till I get flag = True from the function.
        logger.info(f"----------performing dantzig selection---------")
        M_PHI = d
        M_X = 2*d
        NUM_SAMPLES = M_PHI * M_X + M_X

        while True:
            V, function_values, flag = dantzig_selector(oracle, d, rng, epsilon=1e-5, m_phi=M_PHI, m_x=M_X)
            if flag:
                break
            else:
                M_X = int(M_X*2)
                NUM_SAMPLES = M_PHI * M_X + M_X
        logger.info(f"number of samples used in dantzig selector are {NUM_SAMPLES}")     
        directions = V[:m].T
       
        #if np.amax(np.array(function_values)).item() >= finaly:
        #    finaly = np.amax(np.array(function_values)).item()
        #finalx = V[:1]    # TODO will change it later. This is not correct, but anyways we are not using it.
        
    
    #directions = oracle_instance.r[:2].T
    #directions = np.array([[1,0],[0,1],[0,0],[0,0]])
    M = directions.shape[1]   # subspace dimension dimension
    #---------------------------------------------------------
    _, alpha_extent = labelled_corners(directions,bounds)
    # TODO: add the scaling factor later.
    alpha_extent = alpha_extent
    
    #----------------few assertions here--------------------
    angle = np.rad2deg(scipy.linalg.subspace_angles(directions, function_instance.r[:2].T))
    
    logger.info(f"subspace angle is {angle}")
    point_on_estimated_plane, x_star_on_plane, alpha = point_on_plane(function_instance.r[:2], directions.T, x_opt.T,alpha_extent)
    
    logger.info(f"point on the estimated plane is {point_on_estimated_plane}")
    logger.info(f"x_star_on_plane is {x_star_on_plane}")
    logger.info(f"alpha is {alpha}")
    #assert np.isclose(oracle(point_on_estimated_plane.T), fmax), "least squares estimation did not find point on plane"
    #assert x_star_on_plane == True, "x_star is not on the plane, need to increase the domain" 
        
    round_robin = False 
    counter=0       
    t = [{} for _ in range(H_MAX+2)]
    for i in range(H_MAX+2):
        t[i]['alpha_min'] = np.empty((0, M), dtype=float)
        t[i]['alpha_max'] = np.empty((0, M), dtype=float)
        t[i]['alpha_cen'] = np.empty((0, M), dtype=float)
        t[i]['cen_val'] = []

    t[0]['alpha_min'] = alpha_extent[0].reshape(1,-1)
    t[0]['alpha_max'] = alpha_extent[1].reshape(1,-1)
    
    t[0]['alpha_cen'] = ( ((t[0]['alpha_min'] + t[0]['alpha_max'])/2))
    t[0]['cen_val'] = [oracle(  (directions @ t[0]['alpha_cen'].T ).T  )[0]]
    # initilisation of the tree
            
    sampled_value = t[0]['cen_val'][0]    
    
    finaly = sampled_value
    finalx = directions @ t[0]['alpha_cen'].T
    # if sampled_value > finaly:
    #     finalx = directions @ t[0]['alpha_cen'].T
    #     finaly = sampled_value
        
    n = NUM_SAMPLES; regret.append((fmax-finaly))
        
    axis_divisions = {i:0 for i in range(M)}
    for h in np.arange(0,H_MAX+1).reshape(-1):  # we cannot expand the bottom leaves.
                
        n_open_cells = 1 if h == 0 else min(int(C * math.floor(H_MAX/ h)), len(t[h]['alpha_cen']))
        #top_k_tuple = inf_tie_break(t[h]['cen_val'], t[h]['x'], n_open_cells)
        top_k_tuple = top_k(t[h]['cen_val'],n_open_cells)                        
        #-------------------------- NN Look ahead approach ---------------------
        if not round_robin:
            # i-star-cell is the top function representative cell
            i_star_cell_index = top_k_tuple[0][0]
            # Now, Identify the axis to split using look ahead strategy
                    
            minimum_values = []
            
            for axis in range(M):
                # middle cell is the i^star cell
                newmin = t[h]['alpha_min'][i_star_cell_index,:].copy()   # copy the i^{star} cell
                newmax = t[h]['alpha_max'][i_star_cell_index,:].copy()   # copy the i^{star} cell

                # newmin, newmax descirbe the i^{star} cell after dividing along axis
                newmin[axis] = (2 * t[h]['alpha_min'][i_star_cell_index,axis] + t[h]['alpha_max'][i_star_cell_index,axis]) / 3.0
                newmax[axis] = (t[h]['alpha_min'][i_star_cell_index,axis] + 2 * t[h]['alpha_max'][i_star_cell_index,axis]) / 3.0
                
                function_minimum = cell_function_minimum(directions, newmin, newmax, f_hat_numpy,num_points=200)
                minimum_values.append(function_minimum)
            direction_to_split = np.argmax(np.array(minimum_values))
            axis_divisions[direction_to_split] += 1
        else:
            direction_to_split = counter % M
            counter += 1
            axis_divisions[direction_to_split] += 1
        
        logger.info(f"axis_divisions after height {h} are {axis_divisions}")
        #---------------------------end of NN Look ahead--------------------------

        for item in top_k_tuple:
            i_max = item[0]

            # now we have the direction to split, we can split the cell along this direction 
            alpha_cen = t[h]['alpha_cen'][i_max]       # center in the alpha frame
            # alpha_cen_left stores the center of the left box
            alpha_cen_left = alpha_cen.copy()
            alpha_cen_left[direction_to_split] = (5 * t[h]['alpha_min'][i_max,direction_to_split] + t[h]['alpha_max'][i_max,direction_to_split]) / 6.0
            
            # alpha_cen_right stores the center of the right box
            alpha_cen_right = alpha_cen.copy()
            alpha_cen_right[direction_to_split] = (t[h]['alpha_min'][i_max,direction_to_split] + 5 * t[h]['alpha_max'][i_max,direction_to_split]) / 6.0
                            
            # ------------------------left node -----------------------
            t[h + 1]['alpha_cen'] =np.concatenate((t[h+1]['alpha_cen'], alpha_cen_left.reshape(1, -1)), axis=0)
            sampled_value = oracle((directions @ alpha_cen_left).T).item()  # apply transformation
            
            if sampled_value > finaly:
                finalx = directions @ alpha_cen_left
                finaly = sampled_value
            n = n+1
                
            t[h + 1]['cen_val'].append(sampled_value)
            
            regret.append(fmax-finaly)
            print_status(d=d,n=n,h=h,i_max=i_max,x=directions @ alpha_cen_left,value=sampled_value,trees='t1')
            
            # splitting step # left child
            t[h + 1]['alpha_min'] =np.concatenate((t[h + 1]['alpha_min'], t[h]['alpha_min'][i_max].reshape(1,-1)), axis=0)
            
            newmax = t[h]['alpha_max'][i_max].copy()
            newmax[direction_to_split] = (2 * t[h]['alpha_min'][i_max,direction_to_split] + t[h]['alpha_max'][i_max,direction_to_split]) / 3.0
            
            t[h + 1]['alpha_max'] =np.concatenate((t[h + 1]['alpha_max'], newmax.reshape(1,-1)), axis=0)
            
            # ------------------------right node--------------------
            t[h + 1]['alpha_cen'] =np.concatenate((t[h+1]['alpha_cen'], alpha_cen_right.reshape(1,-1)), axis=0)
            sampled_value = oracle((directions @ alpha_cen_right).T).item()    # TODO: change here
            
            if sampled_value > finaly:
                finalx = directions @ alpha_cen_right
                finaly = sampled_value
            n = n+1
            print_status(d=d,n=n,h=h,i_max=i_max,x=directions @ alpha_cen_right,value=sampled_value,trees='t1')
            t[h + 1]['cen_val'].append(sampled_value)
            
            regret.append(fmax-finaly)
            
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

        if not round_robin:
            if (h+1)% 3 ==0:
                if (h+1) >= 100:
                    round_robin = True
                    # need to retrain neural network by collecting the samples on the smaller domain
                    #model = one_layer_net(d, n_hidden, 1)   # instantiate new nn model
                    #optimizer = torch.optim.Adam(model.parameters(),weight_decay=0.5, lr=0.0001)

                # get the i^* cell
                top_k_tuple = top_k(t[h+1]['cen_val'],n_open_cells)
                i_star_cell_index = top_k_tuple[0][0]
                newmin = t[h+1]['alpha_min'][i_star_cell_index,:].copy()
                newmax = t[h+1]['alpha_max'][i_star_cell_index,:].copy()
                # these are the points in alpha space, construct 2^d corners in real space
                
                # Now sampling points on the plane.
                
                alpha_samples = rng.uniform(newmin,newmax,(RETRAINING_NUM_SAMPLES,M))
                data_x = (directions @ alpha_samples.T).T
                data_y = oracle(data_x)
                X = torch.tensor(data_x).type(torch.float64)
                Y = torch.tensor(data_y).type(torch.float64).reshape(-1,1)
                
                
                extent = np.vstack((newmin, newmax))
                two_power_m = list(itertools.product(*extent.T))
                two_power_m_corners = (directions @ np.array(two_power_m).T).T
                
                # each row is a point in two_power_m_corners
            
                #X,Y = sample_random_points(RETRAINING_NUM_SAMPLES,d,oracle,rng=rng,cell =two_power_m_corners)
                #regret.extend([fmax-finaly]*RETRAINING_NUM_SAMPLES)
                #here, need to scale the data
                x_scaler = X_scaler_high_d(two_power_m_corners)
                y_scaler = Y_scaler()
                
                x_transformed = (x_scaler.fit_transform(X)).type(TORCH_DTYPE)
                y_transformed = y_scaler.fit_transform(Y)

                #x_transformed = X
                #y_transformed = Y
                # x_transformed = x_transformed.to('cuda')
                # y_transformed = y_transformed.to('cuda')
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
                #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,momentum=MOMENTUM)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
                logger.info(f"model retraining has started at height {h+1}")
                # model.to('cuda')
                loss = model_train(model,optimizer, x_transformed,y_transformed,3000,scheduler)
                logger.info(f"model retraining has ended at height {h+1} with loss {loss}")
                # model.to('cpu')
                # here, we trained model on the scaled data.

                # since, model is trained on the scaled data, we need to scale the data before sending to the model.
                f_hat_numpy = lambda x:\
                y_scaler.inverse_transform(model(torch.tensor(x_scaler.fit_transform(x.reshape(-1,d))).type(TORCH_DTYPE)).detach().numpy())
                
                #f_hat_numpy = lambda x:\
                #    model(torch.tensor(x).type(TORCH_DTYPE)).detach().numpy()
                
                if d==2:
                    visualize_nn_model(f_hat_numpy,oracle, x_min=-5, x_max=5, h=0, x_scaler =False, y_scaler = False)
        
                average_fit_error = torch.mean((y_transformed - torch.mean(y_transformed))**2)
                
                if model.criterion(model(x_transformed),y_transformed) < average_fit_error:
                    print(f"Model fit is better than average value fit")
                else:
                    print(f"Model fit is not better than average value fit")
                    round_robin = True
        
    return finalx, finaly,regret,x_opt, fmax, x_star_on_plane, alpha, directions, angle, NUM_SAMPLES



