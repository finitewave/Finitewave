finitewave
===========

Package for a wide range of tasks in modeling cardiac electrophysiology using
finite-difference methods.

Package structure
-----------------

core
"""""

Base classes subpackage. Use this subpackage to create your own implementation
and incorporate it in the package logic.

cpuwave2D
"""""""""

Ready-to-use implementation for 2D problems. Contains prepared models,
tissue generation methods, optimized numerical schemes and specialized tools.  

cpuwave3D
"""""""""

Ready-to-use implementation for 3D problems. Contains prepared models, tissue
generation methods, optimized numerical schemes and specialized tools.  

tools
""""""

Additional methods to treat the results, perform statistical analysis and make
different visual representations.
