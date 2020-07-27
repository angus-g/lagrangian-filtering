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

   $ conda install -c conda-forge -c angus-g lagrangian-filtering

to install this package and all its required dependencies. If the
``-c angus-g`` flag is placed before the ``-c conda-forge`` flag, a
modified OceanParcels dependency is pulled in. By default, this
shouldn't be required, but some experimental features may only be
present in this version.

To keep things a bit cleaner, you can install `lagrangian-filtering`
in its own Conda environment:

.. code-block:: console

   $ conda create -n filtering -c conda-forge -c angus-g lagrangian-filtering

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
for `lagrangian-filtering`. In these cases, it is easier to install a modifiable
version of the package in a virtual environment.

.. code-block:: console

   $ git clone https://github.com/angus-g/lagrangian-filtering
   $ cd lagrangian-filtering
   $ virtualenv env
   $ source env/bin/activate
   $ pip install -e .
   $ pip install -r requirements.txt

This will install `lagrangian-filtering` as a
development package, where changes to the files in the git repository
will be reflected in your Python environment. To update `lagrangian-filtering`, run

.. code-block:: console

   $ git pull

In the directory into which you cloned the repository. If the
`parcels` dependency has changes, running

.. code-block:: console

   $ pip install --upgrade --upgrade-strategy eager .

will pull changes to its corresponding git repository.

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

   $ python -m ipykernel install --user --name=filtering

When you're using Jupyter notebooks, you can either change to the new
`filtering` kernel from the `Kernel` menu, or select `filtering`
instead of "Python 3" when creating a new notebook.
