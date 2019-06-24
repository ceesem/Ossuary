from setuptools import setup
import re
import os

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    version='0.0.1',
    name='Ossuary',
    description='Tools for looking at neuronal skeletons',
    author='Casey Schneider-Mizell',
    author_email='caseysm@gmail.com',
    url='https://github.com/ceesem/',
    packages=['ossuary'],
    include_package_data=True,
    install_requires=required,
)

