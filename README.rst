

Welcome to fusilib!
###################
|Zenodo| |License|


What is fusilib?
=================
``fusilib`` is a Python package for the analysis of functional ultrasound and electrophysiology data.

Installation
============

The following package must be installed on your system:

* git
* pip
* python3-venv

To install the ``fusilib`` package in a virtual environment, execute the following from the command line:

.. code-block:: bash

   git clone https://github.com/anwarnunez/fusi.git
   cd fusi
   python3 -m venv fusienv
   source fusienv/bin/activate
   pip install -r requirements.txt
   pip install -e .
  
If you get errors from pip, try re-running the pip install commands.

Getting started
===============

First, make sure everything is installed correctly by importing the package from within Python or IPython:

.. code-block:: python

   >>> import fusilib


Checkout the figshare `project`_ and download the `dataset.zip`_ file (~15GB). Unzip the file in your local machine. Then, open the command line and navigate to the unzipped ``Subjects`` folder. You should then be able to execute the demo scripts from the command line

.. code-block:: bash

   unzip dataset.zip
   cd Subjects
   python3 /path/to/repo/fusi/scripts/demo.py


Alternatively, you can add the location of the downloaded data to the demo `scripts`_ directly. To do so, change the following lines at the top of the demo scripts:

.. code-block:: python   

   # Enter the path to the downloaded "Subjects" directory.
   # By default, the path is set to the current working directory.
   import fusilib.config
   data_location = '/path/to/extracted/data/Subjects'
   fusilib.config.set_dataset_path(data_location)

Then, execute the demo script from the command line:

.. code-block:: bash

   python3 scripts/demo.py

Cite as
=======
Neural correlates of blood flow measured by ultrasound. Nunez-Elizalde AO, Krumin M, Reddy CB, Montaldo G, Urban A, Harris KD, and Carandini M. Neuron (2022). https://doi.org/10.1016/j.neuron.2022.02.012.

   
.. |Zenodo| image:: https://zenodo.org/badge/456774708.svg
   :target: https://zenodo.org/badge/latestdoi/456774708
   
.. |License| image:: https://img.shields.io/badge/license-BSD%203--Clause-blue
   :target: https://opensource.org/licenses/BSD-3-Clause

.. _project: https://figshare.com/projects/Nunez-Elizalde2022/132110

.. _dataset.zip: https://figshare.com/articles/dataset/Simultaneous_functional_ultrasound_and_electrophysiology_recordings_of_neural_activity_in_awake_mice/19316228

.. _scripts: https://github.com/anwarnunez/fusi/tree/main/scripts
