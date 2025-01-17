.. nuPyProp documentation master file, created by
   sphinx-quickstart on Fri Jan 1 15:29:00 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />


Welcome to physicsModels
=======================

physicsModels is a python module containing many atmospheric models for space physics
pertaining to the aurora and high-latitude phenomenon.

The code package contains 4 main modules:-

1. :ref:`AlfvenWave_Resonance`: Model which simulates shear Alfven waves traveling
along high-latitude magnetic field lines to the ionosphere.

2. :ref:`invertedV_fitting`: Model which fits auroral electron differential flux data
to produce ionization profiles and fits to source distributions.

3. :ref:`cross-ionosphere`: Collects attitude slices from IRI and NRLMSIS model
to general altitude profiles of the ionosphere

4. :ref:`magnetosphere_Ionosphere`: Storage for high-altitude profiles of
various plasma parameters.

.. toctree::
   :hidden:
   :caption: Contents:
   :maxdepth: 2

.. AlfvenWave_Resonance.rst
   invertedV_fitting.rst
   ionosphere.rst
   magnetosphere_Ionosphere.rst
