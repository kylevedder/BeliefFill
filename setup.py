# install using 'pip install -e .'

from setuptools import setup

setup(name='BeliefFill',
      packages=['BeliefFill'],
      package_dir={'BeliefFill': 'BeliefFill'},
      install_requires=['torch'],
    version='0.0.1')