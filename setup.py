from setuptools import setup, find_packages
from codecs import open
from os import path

__author__ = "Mathijs Maijer"
__email__ = "m.f.maijer@gmail.com"

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='dynsimf',
      version='0.29',
      license='BSD-Clause-2',
      description='Dynamic Network Simulation Framework',
      url='https://github.com/Tensaiz/DyNSimF',
      author='Mathijs Maijer',
      author_email='m.f.maijer@gmail.com',
      classifiers=[

          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          'License :: OSI Approved :: BSD License',

          "Operating System :: OS Independent",

          'Programming Language :: Python :: 3'
      ],
      keywords='dynamic-networks networks simulator network-of-networks analysis',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test", "dynsimf.test", "dynsimf.test.*"]),
      install_requires=['numpy', 'networkx', 'tqdm', 'pyintergraph', 'python-igraph', 'pillow', 'sphinx-rtd-theme', 'pytest', 'salib', 'matplotlib'],
      )
