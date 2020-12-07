************
Installation
************

To get DyNSimF up and running, multiple choices are available. The easiest way is to use pip, which can be acquired by installing the python ``setuptools``.

================
PIP installation
================

The DyNSimF package lives in the Python Package Index at pypi_. It can be installed by opening a command prompt/terminal and entering:

.. code-block:: python

    pip install dynsimf


This will download and install the necessary requirements and the DyNSimF framework. If you are on Linux, you might have to include ``sudo`` at the beginning of the command.

===================
Source installation
===================

First download the compressed source (e.g: .zip) from pypi_ or GitHub_. Unpack the contents to a desired directory and then run the command:

.. code-block:: python

    python setup.py install to build and install

============
Using GitHub
============

If a copy of the github repository is required, use the git clone command to copy the contents of the repository. 
After cloning, change the directory to the cloned repository and run the setup.py file.

.. code-block:: python

    git clone https://github.com/Tensaiz/DyNSimF
    cd dynsimf
    python setup.py install to build and install


============
Requirements
============

The requirements should be automatically installed when using pip install, 
or manually installing the package using setup.py, but if any are missing, 
this section will indicate which libraries are used and why.



^^^^^
numpy
^^^^^

Most of the core logic to speed up calculations is implemented using numpy. It can also be used to speed up the custom model calculations.


^^^^^^^^
networkx
^^^^^^^^

Networkx is the main library used to represent the graphs internally. 
The graphs are often transformed into numpy adjacency matrices and vice-versa. 

^^^^^^^^^^
matplotlib
^^^^^^^^^^

Used to visualize the simulation outcomes and visualize the network dynamics real-time.

^^^^
tqdm
^^^^

This package is used to show a progress bar when the simulation is running.

^^^^^^^^^^^^
pyintergraph
^^^^^^^^^^^^

A package that supports the transformation between networkx and igraph graphs in python. Used to be able to leverage the visualization options of both libraries

^^^^^^^^^^^^^
python-igraph
^^^^^^^^^^^^^

A ported implementation of the graph library for the R language. It is leveraged in this package to support more visualization options that are only available to igraph.

^^^^^^
pillow
^^^^^^

This package helps with saving the matplotlib animations to a .gif file.

^^^^^^^^^^^^^^^^
sphinx_rtd_theme
^^^^^^^^^^^^^^^^

Used for documentation.

^^^^^^
pytest
^^^^^^

Used for unit testing.

^^^^^
salib
^^^^^

The main package used for sensitivity analysis.


.. _pypi: https://pypi.org/project/dynsimf/
.. _GitHub: https://github.com/Tensaiz/DyNSimF/archive/master.zip