# nsml: nvcr.io/nvidia/pytorch:20.08-py3
from distutils.core import setup
import setuptools

print('setup_test.py is running...')

setup(name='PNS_SAMPLE',
      version='1.0',
      install_requires=['opencv-python']
      ) ## install libraries, 'keras==xx.xx'