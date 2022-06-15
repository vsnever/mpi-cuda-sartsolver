CXX = g++
MPICXX = mpic++
NVCC = nvcc

CXXFLAGS = -Wall -O3 -std=c++17 $(DEFINES)

GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM37    := -gencode arch=compute_37,code=sm_37
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM52    := -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM61    := -gencode arch=compute_61,code=sm_61
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM75    := -gencode arch=compute_75,code=sm_75
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80
GENCODE_SM80    := -gencode arch=compute_86,code=\"sm_86,compute_86\"
GENCODE_FLAGS   := $(GENCODE_SM35) $(GENCODE_SM37) $(GENCODE_SM50) $(GENCODE_SM52) $(GENCODE_SM60) $(GENCODE_SM61) $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM80) $(GENCODE_SM86)

NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets $(DEFINES) $(GENCODE_FLAGS)

SOURCE_DIR = source
KERNEL_DIR = source/cuda
BUILD_DIR = build

ifeq ($(CUDA_HOME),)
CUDA_HOME := /usr/local/cuda
endif
CUDAFLAGS = -I${CUDA_HOME}/include

ifeq ($(HDF5_HOME),)
HDF5_HOME := /usr/lib/x86_64-linux-gnu/hdf5/serial
endif
HDF5FLAGS = -I$(HDF5_HOME)/include

LDFLAGS = -L$(CUDA_HOME)/lib64 -lhdf5 -lhdf5_cpp -lcublas -lcudart

HEADERS = $(SOURCE_DIR)/arguments.hpp $(SOURCE_DIR)/hdf5files.hpp $(SOURCE_DIR)/raytransfer.hpp $(SOURCE_DIR)/laplacian.hpp $(SOURCE_DIR)/sartsolver.hpp $(SOURCE_DIR)/sartsolver_cuda.hpp $(SOURCE_DIR)/image.hpp

TARGETS = sartsolver

OBJS = $(BUILD_DIR)/raytransfer.o $(BUILD_DIR)/laplacian.o $(BUILD_DIR)/image.o $(BUILD_DIR)/sartsolver.o $(BUILD_DIR)/sartsolver_cuda.o $(BUILD_DIR)/sart_kernels.o $(BUILD_DIR)/arguments.o $(BUILD_DIR)/hdf5files.o $(BUILD_DIR)/main.o

all: $(TARGETS)

$(TARGETS): $(OBJS)
	$(MPICXX) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/raytransfer.o: $(SOURCE_DIR)/raytransfer.cpp $(SOURCE_DIR)/raytransfer.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(HDF5FLAGS) -c $(SOURCE_DIR)/raytransfer.cpp -o $@

$(BUILD_DIR)/laplacian.o: $(SOURCE_DIR)/laplacian.cpp $(SOURCE_DIR)/laplacian.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(HDF5FLAGS) -c $(SOURCE_DIR)/laplacian.cpp -o $@

$(BUILD_DIR)/image.o: $(SOURCE_DIR)/image.cpp $(SOURCE_DIR)/image.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(HDF5FLAGS) -c $(SOURCE_DIR)/image.cpp -o $@

$(BUILD_DIR)/sartsolver.o: $(SOURCE_DIR)/sartsolver.cpp $(SOURCE_DIR)/sartsolver.hpp $(SOURCE_DIR)/raytransfer.hpp $(SOURCE_DIR)/laplacian.hpp | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -c $(SOURCE_DIR)/sartsolver.cpp -o $@

$(BUILD_DIR)/sartsolver_cuda.o: $(SOURCE_DIR)/sartsolver_cuda.cpp $(SOURCE_DIR)/sartsolver_cuda.hpp $(SOURCE_DIR)/raytransfer.hpp $(SOURCE_DIR)/laplacian.hpp | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(CUDAFLAGS) -c $(SOURCE_DIR)/sartsolver_cuda.cpp -o $@

$(BUILD_DIR)/sart_kernels.o: $(KERNEL_DIR)/sart_kernels.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $(KERNEL_DIR)/sart_kernels.cu -o $@

$(BUILD_DIR)/arguments.o: $(SOURCE_DIR)/arguments.cpp $(SOURCE_DIR)/arguments.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I $(SOURCE_DIR)/include -c $(SOURCE_DIR)/arguments.cpp -o $@

$(BUILD_DIR)/hdf5files.o: $(SOURCE_DIR)/hdf5files.cpp $(SOURCE_DIR)/hdf5files.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(HDF5FLAGS) -c $(SOURCE_DIR)/hdf5files.cpp -o $@

$(BUILD_DIR)/main.o: $(SOURCE_DIR)/main.cpp $(HEADERS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(HDF5FLAGS) $(CUDAFLAGS) -I $(SOURCE_DIR)/include -c $(SOURCE_DIR)/main.cpp -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(BUILD_DIR)/*.o

cleanall:
	find . -type f -name '*.o' -exec rm {} +
