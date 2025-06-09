import numpy as np
import math
from src.utils.functions import print_status
from src.utils.tie_break import top_k
from src.utils.general import get_logger

harmonic_number = lambda n: sum(1/i for i in range(1, n+1))



def oo(function_instance, n, **kwargs):
    
    logger = get_logger(worker_id=0, model_dir=kwargs.get('save_dir'))
   
    d = kwargs.get('d')

    oracle  = lambda x: -function_instance.evaluate_full(x)
    x_opt = function_instance.x_opt
    fmax = -function_instance.f_opt
      
    regret = []
    

    default_domain = np.vstack((function_instance.lb, function_instance.ub))    
    logger.info(f'The domain of optimization is: {default_domain}')
    
    bounds = kwargs.get('bounds', default_domain)
    # check if the x_opt is inside the domain.
    assert np.all(x_opt >= function_instance.lb) and np.all(x_opt <= function_instance.ub) , 'x_opt is not in the domain of the function'
   
    #SequOOL parameters
    H_MAX = math.floor(n/harmonic_number(n))
    C = 1
    #np_dtype = np.float128
    

    t = [{} for _ in range(H_MAX+2)]
    # t is a list of dictionary.
    # {'x_max': [], 'x_min': [], 'x': [], 'leaf': [], 'new': [], 'cen_val': [], 'bs': [], 'ks': [], 'values': {}}
    for i in range(H_MAX+2):
        t[i]['x_max'] = np.empty((0, d))
        t[i]['x_min'] = np.empty((0, d))
        t[i]['x'] = np.empty((0, d))
        t[i]['cen_val'] = []

    t[0]['x_min'] = bounds[0].reshape(1,-1)   #np.zeros((1, d),dtype=np_dtype)
    t[0]['x_max'] = bounds[1].reshape(1,-1)   #np.ones((1, d),dtype=np_dtype)
    t[0]['x'] =   (t[0]['x_min'] +  t[0]['x_max'] )/2    #    np.full((1, d), 0.5,dtype=np_dtype)


    t[0]['cen_val'] = [oracle(t[0]['x']).item()]

    # ## initilisation of the tree
    ## execution
    finaly, finalx = t[0]['cen_val'][0], t[0]['x']   # stores  the maximum value of a function 'f'
    n = 1; regret.append(fmax-finaly)


    count = 0
    for h in np.arange(0,H_MAX+1).reshape(-1):  # we cannot expand the bottom leaves.

        n_open_cells = 1 if h == 0 else min(int(C * math.floor(H_MAX/ h)), len(t[h]['x']))
        top_k_tuple = top_k(t[h]['cen_val'],n_open_cells) # this returns a list of tuples.

        splitd = count % d
        count += 1
        for item in top_k_tuple:
            i_max = item[0]
            
            xx = t[h]['x'][i_max]
            # x_g stores the centre of the left box.
            x_g = xx.copy()
            x_g[splitd] = (5 * t[h]['x_min'][i_max,splitd] + t[h]['x_max'][i_max,splitd]) / 6.0


            # x_d stores the centre of the right box
            x_d = xx.copy()
            x_d[splitd] = (t[h]['x_min'][i_max,splitd] + 5 * t[h]['x_max'][i_max,splitd]) / 6.0
  

            # left node (center)
            t[h + 1]['x'] =np.concatenate((t[h+1]['x'], x_g.reshape(1, -1)), axis=0)

            # left node (splitting step)
            t[h + 1]['x_min'] =np.concatenate((t[h + 1]['x_min'], t[h]['x_min'][i_max].reshape(1,-1)), axis=0)
            newmax = t[h]['x_max'][i_max].copy()
            newmax[splitd] = (2 * t[h]['x_min'][i_max,splitd] + t[h]['x_max'][i_max,splitd]) / 3.0                    
            t[h + 1]['x_max'] =np.concatenate((t[h + 1]['x_max'], newmax.reshape(1,-1)), axis=0)

            # left node (evaluation step)
            sampled_value = oracle(x_g).item()


            if sampled_value > finaly:
                finalx, finaly = (x_g, sampled_value)
                        
            t[h + 1]['cen_val'].append(sampled_value)
                        
            n = n + 1

            regret.append((fmax-finaly))

            print_status(d=d,n=n,h=h,i_max=i_max,x=x_g,value=sampled_value,trees='t1')
            

            #  right node (center)
            t[h + 1]['x'] =np.concatenate((t[h+1]['x'], x_d.reshape(1,-1)), axis=0)

            # right node (splitting step)
            newmin = t[h]['x_min'][i_max].copy()
            newmin[splitd] = (t[h]['x_min'][i_max,splitd] + 2 * t[h]['x_max'][i_max,splitd]) / 3.0

            t[h+1]['x_min'] = np.concatenate((t[h+1]['x_min'], newmin.reshape(1,-1)),axis=0)
            t[h+1]['x_max'] = np.concatenate((t[h+1]['x_max'],t[h]['x_max'][i_max].reshape(1,-1)), axis=0)

            # right node (evaluation)
 
            sampled_value = oracle(x_d).item()

            if sampled_value > finaly:
                finalx, finaly = (x_d, sampled_value)

            t[h + 1]['cen_val'].append(sampled_value)
            n = n + 1
            print_status(d=d,n=n,h=h,i_max=i_max,x=x_d,value=sampled_value,trees='t1')
#            regret.append(fmax-finaly[0])


            regret.append((fmax-finaly))
                                            

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
 

    return {'finalx': finalx,
            'finaly':finaly,
            'regret':regret, 
            'x_opt': x_opt, 
            'fmax': fmax }



