# install using 'pip install -e .'

from setuptools import setup

setup(name='belief_fill',
      packages=['belief_fill'],
      package_dir={'belief_fill': 'belief_fill'},
      install_requires=['torch', 'numpy', 'joblib'],
    version='0.0.1')