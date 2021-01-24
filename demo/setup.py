#setup.py
from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = ['gcsfs', 'simplejson']
setup(
   name='train',
   version='0.1',
   install_requires=REQUIRED_PACKAGES,
   packages=find_packages(),
   include_package_data=True,
   description='Required for titanic trainer'
)