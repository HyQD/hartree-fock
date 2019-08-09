Getting started
===============
This project is written in pure Python and can therefore be installed using
common package managers.
Note that we not yet released this project to PyPi, and the installation must
therefore be done via Github.



Installation
------------

This project can be installed using ``pip``::

    pip install git+https://github.com/Schoyen/hartree-fock.git

Alternatively, the same task can be accomplished using three commands::

    git clone https://github.com/Schoyen/hartree-fock.git
    cd hartree-fock
    pip install .

This downloads the repository and installs directly from the ``setup.py``-file.
In order to update to the latest version use::

    pip install -U git+https://github.com/Schoyen/hartree-fock.git

or, whilst inside the cloned repo::

    pip install -U .


Pipenv
------

The recommended way to install this project as of now is by using ``pipenv``. Run::

	pipenv install -e git+https://github.com/Schoyen/hartree-fock.git#egg=hartree-fock

This will install the project with all dependencies.


Conda Environment
-----------------

Due to some of the optional dependencies in ``quantum-systems``, it can be
useful to set up a conda environment.
We have included an environment specification file for this purpose::

    conda environment create -f environment.yml
    conda activate hf

Deactivating the ``conda`` environment is done with::

    conda deactivate

The environment can be updated with::

    conda env update -f environment.yml
