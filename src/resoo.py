import numpy as np
from src.utils.functions import print_status
from src.utils.general import get_logger
from src.utils.general import point_on_plane
    

def oo(function_instance, n, seed=0, **kwargs):
    
    logger = get_logger(seed, model_dir=kwargs.get('save_dir'))
    logger.info(f'Running task with seed: {seed}')
    
    # setting the seeds
    rng = np.random.default_rng(seed)
    d = kwargs.get('d')
    m = kwargs.get('m')
    fmax = -function_instance.f_opt
    x_opt = function_instance.x_opt
    oracle = lambda x: -function_instance.evaluate_full(x)
    
    # random plane,
    a = rng.standard_normal((d,d))
    q, _ = np.linalg.qr(a)
    
    subspace_plane = kwargs.get('subspace_plane', q[:, :m])

    
    settings = {
        'nb_iter': n,
        'verbose': 3,
        'sample_when_created': 1,
        'type': 'det',
        'h_max': int(np.ceil(np.sqrt(n))),
        'plotf': lambda x: 0,
        'axis': [0, 1, -3, 3],
        'dim':d,
        "low_dim": m
    }

        
    bounds = kwargs.get('bounds')

    # check if the x_opt is on the plane

    # -------------- few assertions here --------
    #assert np.all(x_opt >= function_instance.lb) and np.all(x_opt <= function_instance.ub) , 'x_opt is not in the domain of the function'

    try:
        point_on_random_plane, x_star_on_plane, alpha = point_on_plane(function_instance.r[:m], 
                        subspace_plane.T, x_opt.T, bounds)
    except:
        point_on_random_plane = None,
        x_star_on_plane = False
        alpha = None

    # if not x_star_on_plane:
    #     return None


    logger.info(f'The point on the estimated plane is: {point_on_random_plane}')
    logger.info(f"x_star_on_plane is {x_star_on_plane}")
    logger.info(f"alpha is {alpha}")

    # if not x_star_on_plane:
    #     logger.info(f'The optimal point is not on the plane. The optimal point is: {x_opt}')
    #     return None

    logger.info(f'The domain of optimization is: {bounds}')     

    t = [{} for _ in range(settings['h_max'])]
    # t is a list of dictionary.
    # {'x_max': [], 'x_min': [], 'x': [], 'leaf': [], 'new': [], 'cen_val': [], 'bs': [], 'ks': [], 'values': {}}
    for i in range(settings['h_max']):
        t[i]['x_max'] = np.empty((0, settings['low_dim']), dtype=float)
        t[i]['x_min'] = np.empty((0, settings['low_dim']), dtype=float)
        t[i]['x'] = np.empty((0, settings['low_dim']), dtype=float)
        t[i]['leaf'] = []
        t[i]['new'] = []
        t[i]['cen_val'] = []

    t[0]['x_min'] = bounds[0].reshape(1,-1)   ##np.zeros((1, d))
    t[0]['x_max'] = bounds[1].reshape(1,-1)   #np.ones((1, d))
    t[0]['x'] = (t[0]['x_min'] +  t[0]['x_max'] )/2 
    t[0]['leaf'] = [1]
    t[0]['new'] = [0]
    # t[0]['cen_val'] = [f(t[0]['x'][0,:])]
 
    t[0]['cen_val'] = [oracle(t[0]['x'] @ subspace_plane.T).item()]
       

    # ## initilisation of the tree
    ## execution
    regret = []
    finaly = t[0]['cen_val'][0]     # stores  the maximum value of a function 'f'
    finalx = t[0]['x'] @ subspace_plane.T   # stores the maximum value of a function 'f'
    at_least_one = 1
    n = 1
    regret.append(max ((fmax-t[0]['cen_val'][0]),0))
    # continue with the rest of the algorithm here
    
    while n < settings['nb_iter']:

        if (at_least_one != 1):
            break
        if (settings['verbose'] > 1):
            pass
            # print('----- new pass %d of %d evaluations used ..\n' % (n,settings['nb_iter']))
        v_max = -float('inf')     # stores the maximum value amongs all the leaves till depth 'd'
        at_least_one = 0
        for h in np.arange(0,settings['h_max']-1).reshape(-1):  # we cannot expand the bottom leaves.
            if n >= settings['nb_iter']:
                break
            i_max = - 1
            b_hi_max = -float('inf')   # b_hi_max is maximum among the leaves at depth "h"
            for i in range(len(t[h]['x'])):
                # Among all leaves of depth h, select (h,i) ∈ argmax(h,j)∈Lt b_h,j(t)
                if ((t[h]['leaf'][i] == 1) and (t[h]['new'][i] == 0)):
                                   
                    b_hi = t[h]['cen_val'][i]

                    if (b_hi > b_hi_max):
                        # computing the maximum b_hi among the leaves 
                        b_hi_max = b_hi
                        i_max = i
                        # i_max stores the index of the maximum at depth h

            if (i_max > - 1): # we found a maximum. open the leaf (h, i_max)
                # if (settings['verbose'] > 2):
                #     pass
                #     # print('max b-value for: %f (%d of %d)..\n' % (b_hi_max,i_max,len(t[h]['x'])-1))
                # # animations (in 1D case only)
                # # if ((settings['verbose'] > 1) and (d == 1)):
                # #     draw_function(0,1,f)
                # #     draw_partition_tree(t,settings)
                #     if (settings['verbose'] > 4):
                #         plt.plot(np.array([t[h]['x_min'][i_max],t[h]['x_max'][i_max]]),np.array([settings.axis(3) + 0.7,settings.axis(3) + 0.7]),'-k','LineWidth',4)
                if (h + 1 > settings['h_max']):
                    if (settings['verbose'] > 3):
                        print('Attempt to go beyond maximum depth refused. \n' % ())
                else:
                    if (b_hi_max >= v_max):
                        at_least_one = 1
                        # sample the state and collect the reward
                        xx = t[h]['x'][i_max]                       
                        t[h]['leaf'][i_max] = 0

                        # this leaf is no longer a leaf
                        # we find the dimension to split # it will be the one with the largest range
                        #splitd = np.argmax(t[h]['x_max'][i_max] - t[h]['x_min'][i_max])
                        #splitd = int(splitd)    # I added -1 here, because of zero indexing. need to validate
                        # xx is the centre of the choosen leaf.

                        # difference = t[h]['x_max'][i_max] - t[h]['x_min'][i_max]
                        # k = np.max(difference)/np.min(difference)
                        # if k >= 1.5:
                        #     splitd = np.argmax(difference)
                        # else:
                        #     splitd = 0
                        splitd = np.argmax(t[h]['x_max'][i_max] - t[h]['x_min'][i_max])
                        splitd = int(splitd)
                        # x_g stores the centre of the left box.
                        x_g = xx.copy()
                        x_g[splitd] = (5 * t[h]['x_min'][i_max,splitd] + t[h]['x_max'][i_max,splitd]) / 6.0
                        
                        # x_d stores the centre of the right box
                        x_d = xx.copy()
                        x_d[splitd] = (t[h]['x_min'][i_max,splitd] + 5 * t[h]['x_max'][i_max,splitd]) / 6.0
                        # splits the leaf of the tree # if dim > 1, splits along the largest dimension # left node
                        
                        # left node                        
                        t[h + 1]['x'] =np.concatenate((t[h+1]['x'], x_g.reshape(1, -1)), axis=0)
                        if settings['sample_when_created']:
                            
                            sampled_value = oracle(x_g @ subspace_plane.T).item() #f(x_g)
                            # sampled_value = f(x_g)
                            if sampled_value > finaly:
                                finalx = x_g @ subspace_plane.T
                                #### --------------------------
                                # in matlab "finalx = xx"
                                ### ---------------------------
                                finaly = sampled_value
                            
                            t[h + 1]['cen_val'].append(sampled_value)
                            
                            n = n + 1
                            regret.append(max(fmax-finaly,0))
                            print_status(d=d,n=n,h=h,i_max=i_max,x=x_g @ subspace_plane.T,value=sampled_value,trees='t1')
                        else:
                            t[h + 1]['cen_val'] = np.array([t[h + 1]['cen_val'],0])                        
                        # splitting step
                        t[h + 1]['x_min'] =np.concatenate((t[h + 1]['x_min'], t[h]['x_min'][i_max].reshape(1,-1)), axis=0)


                        newmax = t[h]['x_max'][i_max].copy()
                        newmax[splitd] = (2 * t[h]['x_min'][i_max,splitd] + t[h]['x_max'][i_max,splitd]) / 3.0
                        
                        t[h + 1]['x_max'] =np.concatenate((t[h + 1]['x_max'], newmax.reshape(1,-1)), axis=0)
                        t[h + 1]['leaf'].append(1)
                        t[h + 1]['new'].append(1)

                        #  right node
                        t[h + 1]['x'] =np.concatenate((t[h+1]['x'], x_d.reshape(1,-1)), axis=0)
                        if settings['sample_when_created']:
                            
                            sampled_value = oracle(x_d @ subspace_plane.T).item()#f(x_d)
                            
                            # sampled_value = f(x_d)
                            if sampled_value > finaly:
                                finalx = x_d @ subspace_plane.T
                                #### --------------------------
                                # in matlab "finalx = xx"
                                ### ---------------------------
                                finaly = sampled_value
                           
                            t[h + 1]['cen_val'].append(sampled_value)

                            n = n + 1
                            regret.append(max(fmax-finaly,0)) 
                            print_status(d=d,n=n,h=h,i_max=i_max,x=x_d @ subspace_plane.T,value=sampled_value,trees='t1')
                        else:                            
                            t[h + 1]['cen_val'] = np.array([t[h + 1]['cen_val'],0])                            
                        # splitting
                        newmin = t[h]['x_min'][i_max].copy()
                        newmin[splitd] = (t[h]['x_min'][i_max,splitd] + 2 * t[h]['x_max'][i_max,splitd]) / 3.0

                        t[h+1]['x_min'] = np.concatenate((t[h+1]['x_min'], newmin.reshape(1,-1)),axis=0)
                        t[h+1]['x_max'] = np.concatenate((t[h+1]['x_max'],t[h]['x_max'][i_max].reshape(1,-1)), axis=0)
                        t[h + 1]['leaf'].append(1)
                        t[h + 1]['new'].append(1)

                        #  central node

                        t[h+1]['x'] = np.concatenate((t[h+1]['x'],xx.reshape(1,-1)),axis=0)            
                        t[h + 1]['cen_val'].append(t[h]['cen_val'][i_max])
                        
                        newmin = t[h]['x_min'][i_max,:].copy()
                        newmax = t[h]['x_max'][i_max,:].copy()
                        ####-------------------------------------------
                        # added splitd will need to check ? Yes.
                        newmin[splitd] = (2 * t[h]['x_min'][i_max,splitd] + t[h]['x_max'][i_max,splitd]) / 3.0
                        newmax[splitd] = (t[h]['x_min'][i_max,splitd] + 2 * t[h]['x_max'][i_max,splitd]) / 3.0

                        t[h+1]['x_min'] = np.concatenate((t[h+1]['x_min'], newmin.reshape(1,-1)),axis=0)
                        t[h+1]['x_max'] = np.concatenate((t[h+1]['x_max'], newmax.reshape(1,-1)),axis=0)
                        t[h + 1]['leaf'].append(1)
                        t[h + 1]['new'].append(1)                

                        # set the max Bvalue and increment the number of iteration
                        v_max = b_hi_max
    # mark old just created leafs as not new anymore
        for h in np.arange(0,settings['h_max']).reshape(-1):
            t[h]['new'] = [0]*len(t[h]['x'])

    # if (settings['verbose'] >1):
    #     if d == 1:
    #         draw_function(0,1,f)
    #         draw_partition_tree(t,settings)

    #     if d==2:
    #         draw_2d_function(1,f,save_dir=save_dir)
    #         draw_2d_partition_tree_sequool(t,settings,save_dir)


    return {
        'finalx': finalx,
        'finaly': finaly,
        'regret': regret,
        'x_opt': x_opt,
        'fmax': fmax
    }

