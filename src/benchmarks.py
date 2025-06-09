# taken from coco repository
import numpy as np
import scipy
import cma
from src.utils.general import get_logger
from src.utils.general import point_on_plane
from skopt import gp_minimize
from bayes_opt import BayesianOptimization

#setting the seeds
class FunctionWrapper:
    def __init__(self, func):
        self.func = func
        self.best_value = float('inf')
        self.evaluations = []

    def __call__(self, x):
        value = self.func(x).item()
        self.evaluations.append(value)
        if value < self.best_value:
            self.best_value = value
        return value

def random_search(function_instance, n, seed=0, **kwargs):
    

    # Setting the seeds
    rng = np.random.default_rng(seed)
    d = kwargs.get('d')
    m = kwargs.get('m', 2)
    int_opt = kwargs.get('int_opt')
    
   
    fmin = function_instance.f_opt
    x_opt = function_instance.x_opt
    f = function_instance.evaluate_full

    # Generate all points at once
    #default_domain = np.vstack((function_instance.lb, function_instance.ub))
    points = rng.uniform(function_instance.lb[0], function_instance.ub[0],size =(n,len(function_instance.ub[0])))
    
    # Evaluate all points and keep track of best values
    best_values = []
    current_best = float('inf')
    
    for x in points:
        value = f(x).item()
        if value < current_best:
            current_best = value
        best_values.append(current_best)
    
    # Convert to numpy array for efficiency
    #best_values = np.array(best_values)
    

    return {
        'returned_value': np.min(np.array(best_values)),
        'fmin': fmin,
        'x_opt': x_opt,
        'best_values': best_values
    }
    
def bo(function_instance, n, seed=0, **kwargs): 
    d = kwargs.get('d')
    fmin = function_instance.f_opt
    x_opt = function_instance.x_opt
    
    def black_box_function(**kwargs):
        x = np.array([kwargs[f'x{i}'] for i in range(d)])
        return -function_instance.evaluate_full(x).item()  # BO maximizes, so negate

        # Define pbounds for each dimension
    pbounds = {f'x{i}': (function_instance.lb[0][i], function_instance.ub[0][i]) for i in range(d)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=0,
        random_state=seed,
    )

    # Calculate init points and iterations
    init_points = min(10, max(1, n // 20))
    n_iter = n - init_points

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )

    # Collect best value at each step (convert maximization to minimization)
    best_values = []
    current_best = float('inf')
    for res in optimizer.res:
        val = -res['target']
        current_best = min(current_best, val)
        best_values.append(current_best)

    # Pad if needed (in case BO stops early)
    if len(best_values) < n:
        best_values += [current_best] * (n - len(best_values))
    best_values = np.array(best_values)

    return {
        'fmin': fmin,
        'x_opt': x_opt,
        'best_values': best_values
    }
    
    

    
    
def direct(function_instance, n,seed=0, **kwargs):
    

    rng = np.random.default_rng(seed)
    d = kwargs.get('d')
    m = kwargs.get('m',2)
    int_opt = kwargs.get('int_opt')
    fmin = function_instance.f_opt
    x_opt = function_instance.x_opt

    # Wrap the evaluation function
    wrapped_func = FunctionWrapper(function_instance.evaluate_full)
    
    bounds = []
    for x,y in zip(function_instance.lb[0], function_instance.ub[0]):
        bounds.append((x,y))

    while True:
        result = scipy.optimize.direct(wrapped_func, 
                    bounds = bounds, 
                    maxfun= n,vol_tol = 1e-128)
        if result.nfev >= n:
            break
        else:
            n = n + (n - result.nfev)

    # Get the best f(x) values during the run
    best_values = np.minimum.accumulate(wrapped_func.evaluations)[:n]


    return {
        'result': result,
        'fmin': fmin,
        'x_opt': x_opt,
        'best_values': best_values
    
    }

def dual_annealing(function_instance, n,seed=0, **kwargs):
    
    #setting the seeds
    rng = np.random.default_rng(seed)
    d = kwargs.get('d')
    m = kwargs.get('m',2)
    int_opt = kwargs.get('int_opt')
    fmin = function_instance.f_opt
    x_opt = function_instance.x_opt
    

    # get bounds from the function class variables
    bounds = function_instance.lb, function_instance.ub


    # we need a tuple of bounds for the scipy optimizer
    bounds = []
    for x,y in zip(function_instance.lb[0], function_instance.ub[0]):
        bounds.append((x,y))
    # Wrap the evaluation function
    wrapped_func = FunctionWrapper(function_instance.evaluate_full)

    result = scipy.optimize.dual_annealing(wrapped_func, 
                    bounds = tuple(bounds),
                    maxfun= n,
                    seed = rng,)

    # Get the best f(x) values during the run
    best_values = np.minimum.accumulate(wrapped_func.evaluations)
    assert best_values.ndim == 1, "best_values should be a 1D array"

    # sometimes, there are more elements. We need to truncate them
    best_values = best_values[:n]

    # return [result, fmin, x_opt]
    return {
        'result': result,
        'fmin': fmin,
        'x_opt': x_opt,
        'best_values': best_values
    }


def cma_es(function_instance, n, random_state, **kwargs):
    d = kwargs.get('d')
   
    
    # Wrap the evaluation function
    def objective(x):
        try:
            return function_instance.evaluate_full(x).item()
        except:
            return function_instance.evaluate_full(x)
    
    # Set up bounds
    bounds = [function_instance.lb[0], function_instance.ub[0]]
    
    
    # Extract an integer seed from the SeedSequence
    if isinstance(random_state, np.random.SeedSequence):
        seed = random_state.generate_state(1, dtype=np.uint32)[0]
    elif isinstance(random_state, int):
        seed = random_state
    else:
        seed = None  # Let CMA-ES generate its own seed



    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(d * [0], 0.5,  # Start at origin with initial step size 0.5
                                  {'maxfevals': n, 
                                   'seed': seed,
                                   'bounds': bounds})
    
    # Run optimization
    best_values = []
    evals = []
    while not es.stop() and es.result.evaluations < n:
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        best_values.append(es.result.fbest)
        evals.append(es.result.evaluations)

        # Interpolate best values to match n
    if len(best_values) > 1:
        interpolated_best_values = np.interp(np.arange(n), evals, best_values)
    else:
        interpolated_best_values = np.full(n, best_values[0])
    
    return {
        # 'result': es.result,
        'fmin': function_instance.f_opt,
        'x_opt': function_instance.x_opt,
        'best_values': interpolated_best_values
    }

# random embedding bayesin optimization
def rembo(function_instance, n, seed=0, **kwargs):
    logger = get_logger(seed, model_dir = kwargs.get('save_dir'))

    rng = np.random.default_rng(seed)
    d = kwargs.get('d')
    m = kwargs.get('m')
    int_opt = kwargs.get('int_opt')
    fmin = function_instance.f_opt
    x_opt = function_instance.x_opt
    oracle = lambda x: function_instance.evaluate_full(x)

    # random plane
    a = rng.standard_normal((d, d))
    q, _ = np.linalg.qr(a)

    subspace_plane = q[:, :m]

    bounds = kwargs.get('bounds')

    try:
        point_on_random_plane, x_star_on_plane, alpha = point_on_plane(function_instance.r[:m], 
                        subspace_plane.T, x_opt.T, bounds)
    except:
        point_on_random_plane = None,
        x_star_on_plane = False
        alpha = None

    if not x_star_on_plane:
        return None

    logger.info(f'The point on the estimated plane is: {point_on_random_plane}')
    logger.info(f"x_star_on_plane is {x_star_on_plane}")
    logger.info(f"alpha is {alpha}")

    if not x_star_on_plane:
        logger.info(f'The optimal point is not on the plane. The optimal point is: {x_opt}')
        return None

    logger.info(f'The domain of optimization is: {bounds}')  

    # Run optimization

    # result = gp_minimize(
    #         func=lambda alpha: objective_function(alpha, directions, objective), 
    #         dimensions = low_d_bounds, 
    #         n_calls=n_eval, 
    #         n_random_starts=10, 
    #         verbose=True)


    # def low_d_high_d_mapping(alpha, directions, objective):
    #     x = alpha @ directions
    #     return objective(x)
    low_d_high_d_mapping = lambda alpha: oracle(subspace_plane @ alpha)[0]

    result = gp_minimize(
            func = low_d_high_d_mapping,
            dimensions = list(zip(bounds[0],bounds[1])),
            n_calls = n,
            n_random_starts = 10,
            verbose = True)

    regret = np.minimum.accumulate(result.func_vals) - fmin


    return {
        'fmin': fmin,
        'x_opt': x_opt,
        'regret': regret
    }

    




