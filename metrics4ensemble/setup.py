import os
from setuptools import setup, find_packages


CLASSIFIERS = [
    'Operating System :: Ubuntu',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Topic :: Scientific/Engineering',
]


setup(name='metrics4ensemble',
      description='Metrics for ensemble to ensemble evaluation',
      long_description=(open('README.rst').read()
                        if os.path.exists('README.rst')
                        else ''),
      version='0.1',
      license='Apache',
      classifiers=CLASSIFIERS,
      author='Clement Brochet',
      author_email='clement.brochet@meteo.fr',
      install_requires=['numpy', 'scipy', 'properscoring'],
      tests_require=[],
      packages=find_packages())
