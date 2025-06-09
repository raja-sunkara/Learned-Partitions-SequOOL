import pickle, os, sys
from pathlib import Path
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# solvers
from src.sequool import oo as sequool
from src.dimension_reduction import stochastic_solver as dimension_reduction
from src.dimension_reduction_random_search import stochastic_solver as dimension_reduction_random_search
#from src.all_dimensions_look_ahead import stochastic_function as all_dimensions_look_ahead
from src.dimension_reduction_and_look_ahead import stochastic_solver as dimension_reduction_and_look_ahead
from joblib import Parallel, delayed

from src.soo import oo as soo
from src.resoo import oo as resoo

#benchmarks
from src.benchmarks import random_search, direct, dual_annealing, cma_es, rembo, bo

# functions
from src.utils.coco_functions import SphereRotated, EllipsoidRotated, RastriginRotated, LinearSlope,\
                                AttractiveSector, StepEllipsoid, Rosenbrock,\
                                Discus, BentCigar, SharpRidge, DifferentPowers,\
                                RastriginRotated, Weierstrass, Schaffers, GriewankRosenbrock,\
                                Schwefel, BraninRotated, Custom_f1, EllipsoidRotated,\
                                StyblinskiTang, Hartmann6

OmegaConf.register_new_resolver("eval", eval)

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)

# Add parent directory (to find 'src/')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Add sibling 'HesBO' directory
hesbo_path = os.path.abspath(os.path.join(project_root, "..", "HesBO"))
if hesbo_path not in sys.path:
    sys.path.insert(0, hesbo_path)


# HesBo code imports
from HesBO.experiments import count_sketch_BO_experiments
from HesBO.experiments import REMBO_separate




# combine all the functions into one list
coco_functions = {
    'sphere_rotated': SphereRotated,'attractive_sector': AttractiveSector,
    'step_ellipsoid': StepEllipsoid,'rosenbrock': Rosenbrock,
    'discus': Discus, 'bent_cigar': BentCigar,'sharp_ridge': SharpRidge,
    'different_powers': DifferentPowers,'rastrigin_rotated': RastriginRotated,
    'weierstrass': Weierstrass, 'schaffers': Schaffers,
    'griewank_rosenbrock': GriewankRosenbrock,'schwefel': Schwefel, 'branin': BraninRotated, 'ellipsoid_rotated': EllipsoidRotated, 
    'custom_f1': Custom_f1,
    'styblinski_tang': StyblinskiTang, 'hartmann6': Hartmann6}


@hydra.main(version_base=None, config_path="./../configs", config_name="run")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir     # hydra saves it here.
    
    
    # Function selection and instance creation
    function = coco_functions[cfg.function_name]
    function_instance = function(d=int(cfg.n_dim), rng=np.random.default_rng(123), 
                                    int_opt = cfg.int_opt, sub_space_dim = int(cfg.low_dim))
        
    # generate the random plane
    rng = np.random.default_rng(seed=cfg.seed)
    #ss = rng.bit_generator._seed_seq
    #child_states = ss.spawn(cfg.num_of_trials)
    numpy_seeds = rng.integers(cfg.numpy_seed.lower_bound, cfg.numpy_seed.upper_bound, size=cfg.num_of_trials)
    torch_seeds = rng.uniform(cfg.torch_seed.lower_bound, cfg.torch_seed.upper_bound, size=cfg.num_of_trials).astype(int)
    
    
    if cfg.method == 'sequool':    
        # update n to include the initial samples
        n = int(cfg.num_exp) + int(cfg.init_samples)

        def run_trial():
            lineage_results = {}
            for budget in np.linspace(1, n+1, cfg.num_lineage_steps, dtype=int)[1:]:
                results = sequool(function_instance, budget,
                              d=int(cfg.n_dim), m=int(cfg.low_dim), int_opt=cfg.int_opt, save_dir=output_dir)
                lineage_results[budget] = results
            return lineage_results

        # no repeated trials. it is a deterministic method.
        results = run_trial()


        # soo method
    elif cfg.method == "soo":
        n = int(cfg.num_exp) + int(cfg.init_samples)
        def run_trial():
            lineage_results = {}
            for budget in np.linspace(1, n+1, cfg.num_lineage_steps, dtype=int)[1:]:
                results = soo(
                    function_instance, budget, d=int(cfg.n_dim), m=int(cfg.low_dim), 
                    int_opt=cfg.int_opt, save_dir=output_dir,
                    cfg=cfg
                )
                lineage_results[budget] = results
            return lineage_results

        results = run_trial()


    elif cfg.method == "direct":
        # this is a deterministic method. no need to run multiple trials.
        n = (int(cfg.num_exp) + int(cfg.init_samples))
        results = direct(function_instance, n,
            d=int(cfg.n_dim), m=int(cfg.low_dim), int_opt=cfg.int_opt)
        


    elif cfg.method == "dual_annealing":
        n = (int(cfg.num_exp) + int(cfg.init_samples))
        results = Parallel(n_jobs=cfg.num_of_trials)(delayed(dual_annealing)(
            function_instance, n, random_state,
            d=int(cfg.n_dim), m=int(cfg.low_dim), int_opt=cfg.int_opt
        ) for random_state, _ in zip(numpy_seeds, torch_seeds))
        

    elif cfg.method == "cma_es":
        n = (int(cfg.num_exp) + int(cfg.init_samples))
        results = Parallel(n_jobs=cfg.num_of_trials)(delayed(cma_es)(
            function_instance, n, random_state,
            d=int(cfg.n_dim), m=int(cfg.low_dim), int_opt=cfg.int_opt
        ) for random_state, torch_seed in zip(numpy_seeds, torch_seeds))
        
    elif cfg.method == 'bayesian_opt':
        n = (int(cfg.num_exp) + int(cfg.init_samples))
        results = bo(function_instance, n,
            d=int(cfg.n_dim), m=int(cfg.low_dim), int_opt=cfg.int_opt)
        
        


    elif cfg.method == "random_search" :
        n = int(cfg.num_exp) + int(cfg.init_samples)
        results = Parallel(n_jobs=cfg.num_of_trials)(delayed(random_search)(
            function_instance, n, random_state,
            d=int(cfg.n_dim), m=int(cfg.low_dim), int_opt=cfg.int_opt
        ) for random_state, torch_seed in zip(numpy_seeds, torch_seeds))

    elif cfg.method == "all_dimensions_look_ahead":
        # just run one trial now, for testing purposes.
        # results = all_dimensions_look_ahead(
        #     function_instance, int(cfg.num_exp), numpy_seeds[0], torch_seeds[0],
        #      d = int(cfg.n_dim), save_dir=output_dir, cfg=cfg
        # )

        n = int(cfg.num_exp) + int(cfg.init_samples)
        results = Parallel(n_jobs=cfg.num_of_trials)(delayed(all_dimensions_look_ahead)(
            function_instance, n, random_state, torch_seed,
            d=int(cfg.n_dim), save_dir=output_dir, cfg=cfg
        ) for random_state, torch_seed in zip(numpy_seeds, torch_seeds))
        
    elif cfg.method == "subspace_random_search":
        n = int(cfg.num_exp) + int(cfg.init_samples)
        def run_trial(random_state, torch_seed):
            results = dimension_reduction_random_search(
                    function_instance, n, random_state, torch_seed,
                    d=int(cfg.n_dim), m=int(cfg.low_dim), int_opt=cfg.int_opt, 
                    save_dir=output_dir, cfg=cfg
                )
            return results
        results = []
        # for random_state, torch_seed in zip(child_states, torch_seeds):
        #     results.append(run_trial(random_state, torch_seed))

        # print(results)
        results = Parallel(n_jobs=cfg.num_of_trials)(
            delayed(run_trial)(random_state, torch_seed) 
            for random_state, torch_seed in zip(numpy_seeds, torch_seeds)
        )
        


    elif cfg.method == "learned_subspace":
       
        # results = Parallel(n_jobs=cfg.num_of_trials)(delayed(dimension_reduction)(
        # function, int(cfg.num_exp), random_state, torch_seed,
        # d=int(cfg.n_dim), m=int(cfg.low_dim), int_opt=cfg.int_opt, save_dir=save_dir,
        #  cfg=cfg) for random_state, torch_seed in zip(child_states, torch_seeds))

        def run_trial(random_state, torch_seed):
            lineage_results = {}
            for budget in np.linspace(1, int(cfg.num_exp) +1, cfg.num_lineage_steps, dtype=int)[1:]:
                results = dimension_reduction(
                    function_instance, budget, random_state, torch_seed,
                    d=int(cfg.n_dim), m=int(cfg.low_dim), int_opt=cfg.int_opt, 
                    save_dir=output_dir, cfg=cfg
                )
                lineage_results[budget] = results
            return lineage_results

        results = []
        # for random_state, torch_seed in zip(child_states, torch_seeds):
        #     results.append(run_trial(random_state, torch_seed))

        print(results)
        results = Parallel(n_jobs=cfg.num_of_trials)(
            delayed(run_trial)(random_state, torch_seed) 
            for random_state, torch_seed in zip(numpy_seeds, torch_seeds)
        )

    elif cfg.method == "dimension_reduction_and_look_ahead":
        results = Parallel(n_jobs=cfg.num_of_trials)(delayed(dimension_reduction_and_look_ahead)(
            function_instance, int(cfg.num_exp), random_state, torch_seed,
            d=int(cfg.n_dim), int_opt=cfg.int_opt, save_dir=output_dir, m=int(cfg.m)
        ) for random_state, torch_seed in zip(child_states, torch_seeds))
        

        
    elif cfg.method == "rembo":
        start_rep = 1
        stop_rep = cfg.num_of_trials
        test_func = cfg.function_name
        total_iter = cfg.num_exp
        low_dim = cfg.low_dim
        high_dim = cfg.n_dim
        initial_n = cfg.init_samples
        variance = cfg.dfo_method_args.hesbo.variance
        
        kern_type = 'Y'
         # function creation
        fmax = -function_instance.f_opt
        x_opt = function_instance.x_opt
        #oracle = lambda x: -function_instance.evaluate_full(x)
        box_size = function_instance.ub[0][0]
        
        class OracleWrapper:
            def __init__(self, function_instance):
                self.function_instance = function_instance

            def evaluate(self, x):
                return -self.function_instance.evaluate_full(x).reshape(-1, 1)
            
            def evaluate_true(self, x):
                return -self.function_instance.evaluate_full(x).reshape(-1,1)
        oracle = OracleWrapper(function_instance)
        
        results = REMBO_separate(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, 
                       total_itr=total_iter, low_dim=low_dim, high_dim=high_dim, 
                       initial_n=initial_n, ARD=True, box_size=box_size, kern_inp_type=kern_type,
                       noise_var=variance, oracle = oracle, fmax = fmax, save_dir=output_dir)
        
        results['fmax'] = fmax
        results['x_opt'] = x_opt
        
        
    elif cfg.method == "hesbo":
        start_rep = 1
        stop_rep = cfg.num_of_trials
        test_func = cfg.function_name
        total_iter = cfg.num_exp
        low_dim = cfg.low_dim
        high_dim = cfg.n_dim
        initial_n = cfg.init_samples
        variance = cfg.dfo_method_args.hesbo.variance
        
        
        # function creation
        fmax = -function_instance.f_opt
        x_opt = function_instance.x_opt
        #oracle = lambda x: -function_instance.evaluate_full(x)
        box_size = function_instance.ub[0][0]
        
        class OracleWrapper:
            def __init__(self, function_instance):
                self.function_instance = function_instance

            def evaluate(self, x):
                return -self.function_instance.evaluate_full(x).reshape(-1, 1)
            
            def evaluate_true(self, x):
                return -self.function_instance.evaluate_full(x).reshape(-1,1)
        oracle = OracleWrapper(function_instance)
        results = count_sketch_BO_experiments(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, 
                                    total_itr=total_iter, low_dim=low_dim, high_dim=high_dim, 
                                    initial_n=initial_n, ARD=True, box_size=box_size, noise_var=variance,
                                    oracle = oracle, fmax = fmax, save_dir=output_dir)
        
        results['fmax'] = fmax
        results['x_opt'] = x_opt
        print(fmax)

    elif cfg.method == "resoo":
        n = int(cfg.num_exp) + int(cfg.init_samples)
        # get a random plane and pass it to the soo function.
        # a = rng.standard_normal((cfg.n_dim, cfg.n_dim))
        # q, _ = np.linalg.qr(a)
        # subspace_plane = q[:, :cfg.low_dim]

        # resoo search domain
        eta = cfg.dfo_method_args.resoo.eta
        # [-m/eta, m/eta]^m
        bounds = np.array([[-cfg.low_dim/eta]*cfg.low_dim, [cfg.low_dim/eta]*cfg.low_dim])
    
        def run_trial(random_state):
            lineage_results = {}
            for budget in np.linspace(1, n+1, cfg.num_lineage_steps, dtype=int)[1:]:
                results = resoo(
                    function_instance, budget, seed = random_state,d=int(cfg.n_dim), m=int(cfg.low_dim), int_opt=cfg.int_opt, 
                    save_dir=output_dir, cfg=cfg, bounds=bounds, low_dim = True
                )
                lineage_results[budget] = results
            return lineage_results

        results = Parallel(n_jobs=cfg.num_of_trials)(
                delayed(run_trial)(random_state) 
                for random_state in numpy_seeds )

    file_name = Path(output_dir) / 'results.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(results, f)      
if __name__ == "__main__":
    main()