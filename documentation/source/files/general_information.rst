
General Information
===================

Pythie is a Python package to postprocess weather forecasts.
Presently, it contains the modules:

* MBM: a module to make member-by-member ensemble forecasts postprocessing based on past (re)forecasts. It implements the methods described in :cite:`gi-van2015ensemble`.

Statement of need
-----------------

TODO

Installation
------------

.. note::

    The present release v0.1alpha of Pythie is a preliminary version for the v0.1 release.
    The release of a Python package is planned for the v0.1 release.

    Pythie is presently compatible with Linux and Mac OS.
    **It is not compatible with Windows for the moment**, but a Windows compatible version will be released soon.

The easiest way to run Pythie for the moment is to use an appropriate environment created through `Anaconda`_.

First install Anaconda and clone the repository: ::

    git clone https://github.com/Climdyn/pythie.git

Then install and activate the Python3 Anaconda environment: ::

    conda env create -f environment.yml
    conda activate pythie

.. note::

    The latter instruction will install a minimal environment for Pythie that is needed for the postprocessing to take place.
    If you want to benefit from the advanced functionalities (plotting, notebooks, etc...), you have to install the extra packages: ::

        conda env update -n pythie -f extra_environment.yml

    Please note that these extra packages are also needed to build the documentation.


Documentation
-------------

To build the documentation, please run (with the conda environment activated): ::

    cd documentation
    make html


You may need to install `make`_ if it is not already present on your system.
Once built, the documentation is available `here <../index.html>`_.

You may also want to run the tests provided inside the documentation: ::

    make doctest

In case of a test failure, please report it on the issue page of the Pythie GitHub repository.

.. warning::

    You need the extra packages to be installed in order to be able to build the documentation.

Examples
--------

Examples are provided in the :ref:`files/references:References` pages and through the use of Jupyter notebooks.
The latter can be found in the `notebooks folder <../../../../notebooks>`_.

These notebooks needs the download of a dataset stored on `Zenodo <https://zenodo.org/record/4707154#.YIAvXBI69Go>`_ .
This dataset can be downloaded by running the following script: ::

    bash -i download_sample_data.sh

in the root folder of Pythie. The data are placed by the download script in the :code:`./data` folder.

From there, running ::

    conda activate pythie
    cd notebooks
    jupyter-notebook

will lead you to your favorite browser where you can load and run the examples.

.. warning::

    Beware of your internet connection usage! The size of the example dataset is roughly 2 gigabytes!

Dependencies
------------

Pythie needs mainly:

* Python >= 3.7
* `Numpy`_ for numeric support
* `Scipy`_ for the CPRS minization

Check the yaml file `environment.yml <../../../../environment.yml>`_ for the full list of dependencies.

Forthcoming developments
------------------------

* Better interfacing with libraries such as: `pandas`_, `iris`_, `xarray`_.
* Release of a Python package for the version v0.1
* A more involved test framework

Contributing to Pythie
----------------------

Writing better interfaces to other libraries is a priority so if you think you can help us, please contact the main authors.

In addition, if you have made changes that you think will be useful to others, please feel free to suggest these as a pull request on the Pythie GitHub repository.

A review of your pull request will follow with possibly suggestions of changes before merging it in the master branch.
Please consider the following guidelines before submitting:

* Before submitting a pull request, double check that the branch to be merged contains only changes you wish to add to the master branch. This will save time in reviewing the code.
* Please document the new functionalities in the documentation. Code addition without documentation addition will not be accepted. The documentation is done with `sphinx`_ and follows the Numpy conventions. Please take a look to the actual code to get an idea about how to document the code.
* The team presently maintaining qgs is not working full-time on it, so please be patient as the review of the submission may take some time.

Reporting issues with the software and getting support
------------------------------------------------------

Issues can be reported and support can be asked directly on the Pythie GitHub repository `issues page`_.
However, please be patient as the Pythie team is quite small.

References
----------

.. bibliography:: ref.bib
    :keyprefix: gi-

.. _Numpy: https://numpy.org/
.. _Scipy: https://www.scipy.org/scipylib/index.html
.. _pandas: https://pandas.pydata.org/
.. _iris: https://scitools.org.uk/
.. _xarray: http://xarray.pydata.org/en/stable/index.html
.. _multiprocessing: https://docs.python.org/3.7/library/multiprocessing.html#module-multiprocessing
.. _Anaconda: https://www.anaconda.com/
.. _make: https://www.gnu.org/software/make/
.. _sphinx: https://www.sphinx-doc.org/en/master/
.. _unittest: https://docs.python.org/3/library/unittest.html
