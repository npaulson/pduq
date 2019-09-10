======
Set-up
======

We have based our PDUQ examples on the `Cu-Mg example <http://espei.org/en/latest/cu-mg-example.html>`_
from ESPEI, so please look at that first if you want to understand
what is required to use PDUQ for your own system. In short,
ESPEI takes the raw data, Gibbs energy models, and phase descriptions
and computes the parameter values for those models. What makes ESPEI
different from traditional CALPHAD tools is that it finds the parameters
through Bayesian methods and can therefore provide a distribution for
each parameter instead of a single, deterministic value. This enables
all of the uncertainty quantification tools that PDUQ provides.

In all of our examples, we will be using the outputs of the `trace.npy`
file. `trace.npy` simply contains all of the Gibbs energy parameter
sets generated from a single ESPEI run. `trace.npy` has three dimensions,
the first for the number of "walkers" in the MCMC algorithm ESPEI uses,
the second for the number of total MCMC iterations, and the last for
the number of paramters.

As an example, one trace file might have the following shape

.. code-block:: python

    import numpy as np
    print(np.load('trace.npy').shape)

.. parsed-literal::

    (150, 350, 15)

meaning that there are 150 walkers, 350 iterations, and 15 parameters.
We will be using this trace file along with the `CU-MG_param_gen.tdb`
file for the remaining examples.

