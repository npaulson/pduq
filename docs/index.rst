.. pduq documentation master file, created by
   sphinx-quickstart on Sun Aug 18 10:57:17 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PDUQ
====

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   Installation <installation>
   Examples <examples>
   API Documentation <api>
   Release History <release-history>

PDUQ, or Phase Diagram Uncertainty Quantification, is a tool
to quantify the uncertainty of thermochemistry and phase stability
predictions from the CALPHAD methodology. PDUQ relies on
`pycalphad <https://pycalphad.org/docs/latest/>`_ for property
prediction and `ESPEI <http://espei.org/en/latest/index.html>`_ for
Bayesian samples from the posterior CALPHAD parameter distributions.


License
=======

PDUQ has a BSD-3 Open Source License.

.. code-block:: none

   Copyright (c) 2019, Argonne National Laboratory
   Laboratory. All rights reserved.
   
   Software Name: Phase Diagram Uncertainty Quantification (PDUQ)
   By: Argonne National Laboratory, Pennsylvania State University
   and Jet Propulsion Laboratory, California Institute of
   Technology
   OPEN SOURCE LICENSE
   
   Redistribution and use in source and binary forms, with or
   without modification, are permitted provided that the
   following conditions are met:
   
   1. Redistributions of source code must retain the above
      copyright notice, this list of conditions and the
      following disclaimer.
   2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials
      provided with the distribution.
   3. Neither the name of the copyright holder nor the names of
      its contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.


   **************************************************************
   DISCLAIMER
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
   NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
   **************************************************************


Citing PDUQ
===========

If you use PDUQ for work presented in a publication, we ask that
you cite the following publication:

N.H. Paulson, B.J. Bocklund, R.A. Otis, Z.-K. Liu, M. Stan,
Quantified uncertainty in thermodynamic modeling for materials
design, Acta. Mat. (2019) Vol 174, 9-15.
doi:`10.1016/j.actamat.2019.05.017 <https://doi.org/10.1016/j.actamat.2019.05.017>`_

.. code-block:: none

   @article{PAULSON20199,
            title = "Quantified uncertainty in thermodynamic modeling for materials design",
            journal = "Acta Materialia",
            volume = "174",
            pages = "9 - 15",
            year = "2019",
            issn = "1359-6454",
            doi = "https://doi.org/10.1016/j.actamat.2019.05.017",
            url = "http://www.sciencedirect.com/science/article/pii/S1359645419302915",
            author = "Noah H. Paulson and Brandon J. Bocklund and Richard A. Otis and Zi-Kui Liu and Marius Stan",
            keywords = "CALPHAD, Probability and statistics modeling, Materials design, Metastable phases"
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
