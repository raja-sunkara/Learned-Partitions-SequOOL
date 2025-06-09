# from setuptools import find_packages, setup


# setup(
#     name="bb_optimization",
#     version="0.1",
#     packages=find_packages(where="src"),
#     package_dir={"": "src"},
# )

from setuptools import setup, find_packages

setup(
    name='bb-optimization',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'scipy==1.12.0',
        'matplotlib==3.10.3',
        'pandas==2.3.0',
        'torch==2.7.1',
        'scikit-learn==1.7.0',
        'hydra-core==1.3.2',
        'omegaconf==2.3.0',
        'hydra-joblib-launcher==1.2.0',
        'bayesian-optimization==2.0.4',
        'scikit-optimize==0.10.2',
        'gpy==1.13.2',
        'pyDOE==0.3.8',
        'ray==2.46.0',
        'joblib==1.5.1',
        'shapely==2.1.1',
        'tensorboardx==2.6.2.2',
        'cvxpy==1.6.5',
        'cma==4.2.0',
        "tqdm==4.67.1",
    ],
    entry_points={
        'console_scripts': [
            # Optional: add CLI commands if needed
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
