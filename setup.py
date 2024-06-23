# D-CODE/setup.py
from setuptools import setup, find_packages

setup(
    name='D-CODE',
    version='0.1',
    packages=find_packages(),
    install_requires=['scikit-learn>=0.22.1',
                        'joblib>=0.13.0']
)