Parallel CUDA-accelerated SART solver for large dense matrices
==============================================================

Implementation of the constrained SART solver (Simultaneous Algebraic Reconstruction Technique)
optimized for large dense matrices (for example, when the matrix also contains reflected
radiation data). The solver is parallelized with MPI and accelerated with Nvidia CUDA.
The matrix is distributed between MPI processes and GPUs, which allows the solver to work
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

Note:
Remove unsupported CUDA architectures from GENCODE_FLAGS for older/newer CUDA toolkit versions.

On Ubuntu:
```bash
sudo apt install libhdf5-dev
sudo apt install nvidia-cuda-toolkit
sudo apt install libopenmpi-dev
git clone https://github.com/vsnever/mpi-cuda-sartsolver
git clone https://github.com/p-ranav/argparse
cd mpi-cuda-sartsolver
ln -s -- "$PWD/../argparse/include/argparse" source/include/argparse
make
```

On ITER SDCC:
```bash
module load HDF5/1.10.7-GCCcore-10.2.0-serial
module load CUDA/11.1.1-GCC-10.2.0
module load OpenMPI/4.1.0-GCC-10.2.0
git clone https://github.com/vsnever/mpi-cuda-sartsolver
git clone https://github.com/p-ranav/argparse
cd mpi-cuda-sartsolver
ln -s -- "$PWD/../argparse/include/argparse" source/include/argparse
make
```
