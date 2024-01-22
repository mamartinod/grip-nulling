#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='grip',
    version='1.0.0',
    author='M.-A. Martinod',
    description='Self-calibration data reduction tools for nulling interferometry',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'h5py',
        'emcee',
        'itertools',
        'timeit',
        'cupy'
    ],
)