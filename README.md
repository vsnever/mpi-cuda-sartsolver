Parallel CUDA-accelerated SART solver for large dense matrices
==============================================================

Implementation of the limited SART solver (Simultaneous Algebraic Reconstruction Technique)
optimized for large dense matrices (for example, when the matrix also contains reflected
radiation data). The solver is parallelized with MPI and accelerated with Nvidia CUDA.
The matrix is ​​distributed between MPI processes and GPUs, which allows the solver to work
with matrices that exceed the amount of RAM of individual GPUs and compute nodes.

The solver uses the HDF5 format for input and output data.

Currently, the solver uses CUDA-unaware MPI.

Dependencies
------------

- Any modern MPI implementation
- [HDF5 C/C++ API](https://github.com/HDFGroup/hdf5)
- [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [argparse](https://github.com/p-ranav/argparse)

Installation
------------

On Ubuntu:
```bash
sudo apt install libhdf5-dev
sudo apt install nvidia-cuda-toolkit
sudo apt install libopenmpi-dev
git clone https://github.com/vsnever/mpi-cuda-sartsolver
git clone https://github.com/p-ranav/argparse
cd mpi-cuda-sartsolver
ln -s ../argparse/include/argparse source/include/argparse
make
```

