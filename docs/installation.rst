============
Installation
============

Currently, PDUQ must be installed from source. First, install the dependencies::

    $ conda install -c conda-forge -c pycalphad 'pycalphad>=0.8' numpy scipy 'sympy>=1.2' six 'dask>=0.18' distributed 'tinydb>=3.8' scikit-learn emcee seaborn espei

Then install pduq from the command line as follows::

    $ git clone https://github.com/npaulson/pduq.git
    $ cd pduq
    $ pip install .

NOTE: The following functionality will be available shortly with the first release

At the command line::

    $ pip install pduq
