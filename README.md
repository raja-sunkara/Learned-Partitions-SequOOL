
Source code and evaluation scripts for our <i>ICML 2025</i> paper:

## Adaptive Partitioning Schemes for Optimistic Optimization
[Link to paper]()<br>
[arXiv]()

### Abstract
Applications such as engineering design often require us to optimize a black-box function, i.e., a system whose inner processing is not analytically known and whose gradients are not available. Practitioners often have a fixed budget for the number of function evaluations and the performance of an optimization algorithm is measured by its simple regret. In this paper, we study the class of ``Optimistic Optimization'' algorithms for black-box optimization that use a partitioning scheme for the domain. We develop algorithms that learn a good partitioning scheme and use flexible surrogate models such as neural networks in the optimization procedure. For multi-index functions on an $m$-dimensional subspace within $d$ dimensions, our algorithm attains $\tilde{O}(n^{-\beta / d})$ regret, where $\beta = 1 + \frac{d-m}{2m-1}$, as opposed to $\tilde{O}(n^{-1/d})$ for SequOOL, a state-of-the-art optimistic optimization algorithm. We use our approach to improve the quality of Activation-aware Weight Quantization (AWQ) of the OPT-1.3B model, achieving $\sim10$\% improvement in performance relative to the best possible unquantized model.




## 🛠️ Installation & Setup

This repository is compatible with **Python 3.10**. We recommend using a clean Conda environment to avoid dependency conflicts.

### 1. Clone the repository

```bash
git clone https://github.com/raja-sunkara/Learned-Partitions-SequOOL.git
cd Learned-Partitions-SequOOL
```

### 2. Create and activate a new Conda environment

```bash
conda create -n bbopt python=3.10 -y
conda activate bbopt
```

### 3. Install the package and dependencies

This project uses a `setup.py` file with pinned versions for compatibility:

```bash
pip install -e .
```

This installs all required packages, including:

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `torch`, `joblib`, `hydra-core`, `hydra-joblib-launcher`, `omegaconf`
- `gpy`, `pyDOE`, `cvxpy`, `shapely`
- `scikit-learn`, `scikit-optimize`, `bayesian-optimization`
- `cma`, `ray`, `tqdm`



---

## 🚀 Running Experiments

To run a single optimization:

```bash
python3 scripts/main.py method=METHOD_NAME function_name=FUNCTION_NAME
```

Example:

```bash
python3 scripts/main.py method=sequool function_name=rastrigin_rotated
```

To run multiple methods in parallel using Hydra:

```bash
python3 scripts/main.py --multirun method=rembo,hesbo,direct function_name=branin
```

Hydra will create output logs and results under `experiments/hydra_outputs/`.

---

## 🧪 Supported Algorithms

- Direct Search
- CMA-ES
- Dual Annealing
- Random Search
- Bayesian Optimization
- REMBO
- HESBO
- SOO / RESOO
- SequOOL
- SequOOL on learned subspace

---






Our proposed algorithm is SequOOL on \hat{A} and  Benchmarka are, Direct, SequOOL, Dual Annealing, RESOO, HESBO, SOO, CMA-ES, REMBO, Bayesian Optimization, Random Search.

- Direct (benchmarks)
- SequOOL (sequool_nn)
- Dual Annealing (benchmarks)
- RESOO (resoo)
- HESBO 
- SOO (soo)
- CMA-ES (benchmarks)
- SequOOL on \hat{A} (dimension_reduction)
- REMBO (benchmarks)
- Random Search (dimension_reduction_random_search)
- Bayesian Optimization (benchmarks)








python3 scripts/coco_test_f_main_hydra.py --multirun method=hesbo,rembo2 function_name=rosenbrock hydra/launcher=joblib


python3 main.py --multirun method=dimension_reduction,resoo,rembo,sequool,direct,dual_annealing,random_search,cma_es,soo,hesbo,bayesian_opt function_name=branin hydra/launcher=joblib




python3 scripts/coco_test_f_main_hydra.py --multirun method=rembo2,hesbo function_name=hartman6,styblinski_tang,rosenbrock hydra/launcher=joblib

python3 scripts/coco_test_f_main_hydra.py --multirun function_name=sharp_ridge,rastrigin_rotated,sphere_rotated,different_powers,rosenbrock,styblinski_tang,hartmann6,branin,ellipsoid_rotated,custom_f1,sharp_ridge hydra/launcher=joblib method=dimension_reduction,resoo,rembo2,sequool,direct,dual_annealing,random_search,cma_es,soo,hesbo




