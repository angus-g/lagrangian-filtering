==============
 Installation
==============

There are two ways to install the library, depending on your Python
packaging ecosystem of choice. If you're unsure, the Conda approach is
the most plug-and-play, getting you up and running as easily as
possible.

Using Conda
-----------

If you use Conda_ to manage Python packages, you may run

.. code-block:: console

   $ conda install -c angus-g -c conda-forge lagrangian-filtering

to install this package and all its required dependencies. The ``-c
angus-g`` flag must come before the ``-c conda-forge`` flag to ensure
the correct OceanParcels dependency is pulled in.

To keep things a bit cleaner, you can install `lagrangian-filtering`
in its own Conda environment:

.. code-block:: console

   $ conda create -n filtering -c angus-g -c conda-forge lagrangian-filtering

The created environment can be activated by running

.. code-block:: console

   $ conda activate filtering

And deactivated by running

.. code-block:: console

   $ conda deactivate

.. _Conda: https://conda.io

Using Pip
---------

On the other hand, you may not use Conda, or you wish to develop
either `lagrangian-filtering` or its `modified OceanParcels
dependency`_. In these cases, it is easier to install a modifiable
version of the package in a virtual environment.

.. code-block:: console

   $ git clone https://github.com/angus-g/lagrangian-filtering
   $ cd lagrangian-filtering
   $ virtualenv env
   $ source env/bin/activate
   $ pip install -r requirements.txt
   $ pip install -e .

This will install both `lagrangian-filtering` and `parcels` as
development packages, where changes to the files in the git repository
will be reflected in your Python environment. To update `lagrangian-filtering`, run

.. code-block:: console

   $ git pull

In the directory into which you cloned the repository. If the
`parcels` dependency has changes, running

.. code-block:: console

   $ pip install -r requirements.txt

will pull changes to its corresponding git repository.

.. _modified OceanParcels dependency: https://github.com/angus-g/parcels

Working with Jupyter Notebooks
------------------------------

If you're working with Conda environments, or a regular virtual
environment, it may be the case that you install
`lagrangian-filtering`, but ``import filtering`` fails within a Jupyter
notebook. This is because Jupyter doesn't know about your environment,
so it's likely looking at your system Python installation instead. We
can fix this by adding a new *kernel*. These instructions will be
specific to pip, but you can substitute the activation and
installation commands for Conda. First, make sure your environment is
activated:

.. code-block:: console

   $ source env/bin/activate

Now install `ipykernel`

.. code-block:: console

   $ pip install ipykernel

You can use this package to register a new kernel for your environment:

.. code-block:: console

   $ python -m ipykernel --user --name=filtering

When you're using Jupyter notebooks, you can either change to the new
`filtering` kernel from the `Kernel` menu, or select `filtering`
instead of "Python 3" when creating a new notebook.
