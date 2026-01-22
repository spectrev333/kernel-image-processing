# GPU-accellerated Kernel Image Processing
This repo contains the code for the second assignment of the Parallel Programming for Machine Learning course.

## Abstract
This project aims to compare a naive sequential al-
gorithm for 2D image convolution kernels with a GPU-
accelerated implementation. The parallel implementation
was developed using the AMD ROCm platform and the HIP
API. Using common GPU optimisation techniques, such as
shared memory tiling and constant memory caching for fil-
ter coefficients, the GPU implementation achieved a signif-
icant speedup compared to the naive sequential baseline.

[Report](Kernel_Image_Processing.pdf)