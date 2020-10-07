# manifold GPLVM (mGPLVM)

This directory contains the data and code used for the 2020 NeurIPS paper *Manifold GPLVMs for discovering non-Euclidean latent structure in neural data*.

A simple example calculation can be found in *example.jl*

Here is a brief overview of the remaining codebase

#### src/
Code for generating data and running mGPLVM calculations.\
*fit_gplvm.jl* - general fitting function for mGPLVM models.\
*gplvm_utils.jl* - function for calculating ELBOs and calling various manifold-specific helper functions.\
*distance_functions_gp* - functions for computing kernels and logP(Y|{g}).\
*fitting_data.jl* - code for generating synthetic data.

#### example_code/
Code for fitting example models in the paper.\
Separate scripts for fitting example mGPLVMs to fly data, S^3, SO(3) and T^2.\
S^2 is treated in a separate OCaml implementation.

#### cv_code/
Code for running cross-validation calculations.\
*comparative_fly_cv.jl* - code for running cross-validations on fly data.\
*comparative_gplvm_cv.jl* - code for running cross-validations on T2 and SO3 vs R2/R3.\
*comparative_threefold_cv.jl* - code for running threefold crossvalidation on T2/SO3/S3.\
*comparative_cv_analysis.jl* - code for computing MSEs and test log likelihoods.

#### analysis_code/
Various scripts used for analyses after fitting models.\
*calc_LL.jl* - used to compute importance weighted log likelihoods (Burda 2015).

#### figure_code/
Code for generating figures 3-4 in the paper as well as supplementary figures on ARD and direct products.

#### fly_data/
Contains the dataset used for the *Drosophila* analyses.

#### results/
Contains results of all the model fits as .jld files.

#### figures/
Contains all the subfigures generated from *figure_code/*.\



### Implementation details

A full specification of packages used and their versions can be found in *packages.txt*.

Results were found not to depend strongly on learning rates, and learning rates of 0.01-0.05 were used as a balance between rate of convergence and stability. In general, smaller learning rates were used for fitting larger models.

For all calculations, a 'burn-in' period was used where variational covariances were fixed and no entropy was included in the ELBO, specified by the 'minH' input to *fit_mgplvm*. This was found to be particularly important in periodic spaces to avoid an early collapse of the variational distribution. In general, longer burn-in periods were needed for higher-dimensional problems and we used values of 100 (1D) to 300 (2D/3D) for our calculations.

Variational distributions were initialized at the identity of the group with a covariance matrix spanning the latent space (see *initialize_parameters.jl* for details). Inducing points were initialized randomly according to the prior for each manifold.

All calculations were run on a single CPU and took on the order of minutes to a couple of hours (see paper appendix for details on complexity and computation).
A more efficient implementation using PyTorch and parallelization across GPUs is currently under development for working with larger datasets and will be released when ready.
This will also include general-purpose methods for direct product kernels and ARD which were hard-coded for the examples in the paper.
The PyTorch implementation was also used to run the computations on the mouse head direction data.

Alignment functions for S3 and SO(3) as well as the S2-mGPLVM were implemented in OCaml and are not included here (see paper appendix for details on alignment).
