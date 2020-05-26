""" Install package script. """
from setuptools import setup, find_packages


setup(name='baselines',
      packages=[package for package in find_packages()
                if package.startswith('baselines')],
      install_requires=[
          'gym',
          'scipy',
          'tqdm',
          'joblib',
          'dill',
          'progressbar2',
          'cloudpickle',
          'click',
          'opencv-python'
      ],
      version='0.0.0')