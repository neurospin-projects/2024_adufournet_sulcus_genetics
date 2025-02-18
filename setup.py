import os
from setuptools import setup, find_packages

setup(
    name='2024_adufournet_sulcus_genetics',
    version='0.0.1',
    packages=find_packages(
        exclude=['notebooks*']),
    license='CeCILL license version 2',
    description='Deep learning models '
                'to analyze anterior cingulate sulcus patterns',
    long_description=open('README.rst').read(),
    install_requires=['pandas',
                      'scipy',
                      'matplotlib',
                      'tqdm',
                      'scikit-learn>=0.24',
                      'plotly',
		              'ipykernel',
                      'seaborn',
                      'statsmodels',
                      'umap-learn',
                      'numpy',
                      'plotly',
                      ],
    extras_require={"anatomist": ['deep_folding @ \
                        git+https://git@github.com/neurospin/deep_folding',
                      ],
    },
    url='https://github.com/neurospin-projects/2024_adufournet_sulcus_genetics',
    author='Antoine Dufournet, Julien Laval, Joël Chavas, Vincent Frouin',
    author_email='antoine.dufournet@cea.fr, julien.laval@cea.fr, joel.chavas@cea.fr, vincent.frouin@cea.fr'
)
