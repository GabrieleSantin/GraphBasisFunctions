# GraphBasisFunctions

Python code for Graph Basis Functions approximation.

This code is a (partial) porting of the Matlab toolbox [GBFlearn](https://github.com/WolfgangErb/GBFlearn) writtten by [Wolfgang Erb](https://www.lissajous.it/).

The implementation structure and the code of the greedy algorithms is taken from the [VKOGA](https://github.com/GabrieleSantin/VKOGA) package.

All the approximants are implemented as [scikit-learn](https://scikit-learn.org/stable/) `Estimator`, and thus they can be combined with the parameter 
optimization tools of the package.



## Quick start

You can start with one of the demos:
* [demo_interpolation.ipynb](demo_interpolation.ipynb): An introduction to the basic computation of a GBF approximant.
* [demo_error_decay.ipynb](demo_error_decay.ipynb): A comparison of the decay of the approximation error for different kernels.
* [demo_parameter_optimization.ipynb](demo_parameter_optimization.ipynb): An introduction to the use of `sklearn` to optimize the paramters.
* [demo_interpolation_fgreedy.ipynb](demo_interpolation_fgreedy.ipynb): An introduction to the basic computation of a GBF approximant via f-greedy.
* [demo_point_selection_Pgreedy.ipynb](demo_point_selection_Pgreedy.ipynb): An introduction to the point selection via P-greedy.
* [demo_point_selection_Pgreedy_par_optimization.ipynb](demo_point_selection_Pgreedy_par_optimization.ipynb): Point selection via P-greedy with optimization of parameters.


## Code overview

The code is organized as follows:
* [approx.py](approx.py): Definition of the GBF approximation models (GBFIntepolation, GBFGreedy).
* [kernels.py](kernels.py): Definition of the graph kernels and of the GBFs.
* [utils.py](utils.py): Definition of various utility functions.
* [graph_loaders.py](graph_loaders.py): Definition of various utility to load some example graphs.
