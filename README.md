
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
















