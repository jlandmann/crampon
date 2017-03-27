# -*- coding: utf-8 -*-
"""
This setup file for crampon was modified from the PyPA sample. For more options
 in the setup see there.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""


from setuptools import setup, find_packages  # Always setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path


here = path.abspath(path.dirname(__file__))

NAME = 'CRaMPON'
DESCRIPTION = 'A Python Project for Cryospheric Monitoring and Prediction ' \
              'Online'
LONG_DESCRIPTION = 'The long description is still to come...'
AUTHOR = 'Johannes Landmann and others'
AUTHOR_EMAIL = 'landmann@vaw.baug.ethz.ch'
URL = 'https://github.com/jlandmann/CRaMPON'
KEYWORDS = ['glacier', 'mass-balance', 'runoff', 'model', 'python',
            'monitoring', 'prediction']


# Get the long description from the README file
with open(path.join(here, NAME.lower(), 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


with open('LICENSE') as f:
    LICENSE = f.read()


# Versions should comply with PEP440.  For a discussion on single-sourcing
#  the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html
# version/release string
# This can be replaced e.g. by something similar to
# https://github.com/fabric/fabric/blob/master/fabric/version.py
MAJOR = 0
MINOR = 0
MINI = 1
VERSION = '{!s}{!s}{!s}'.format(MAJOR, MINOR, MINI)


setup(
    name=NAME,
    version=VERSION,

    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,

    author=AUTHOR,
    author_email=AUTHOR_EMAIL,

    license=LICENSE,

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        # 'License :: Other/Proprietary License',

        # Python versions supported
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    keywords=" ".join(KEYWORDS),
    packages=find_packages(exclude=('tests', 'docs'))
)
