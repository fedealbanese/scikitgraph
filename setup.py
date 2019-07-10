from setuptools import setup
from setuptools import find_packages

setup(name='scikitgraph',
      version='1.0',
      description='scikit learn extention for graphs',
      author='me',
      author_email='ffalbanese@gmail.com',
      url='https://github.com/fedealbanese/scikitgraph',
      download_url='https://github.com/fedealbanese/scikitgraph',
      license='MIT',
      install_requires=['numpy',
                        'networkx',
                        'sklearn',
                        'pandas',
                        'node2vec'
                        ],
      package_data={'all_data': ['README.md']},
packages=find_packages())
