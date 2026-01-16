# CUDA Makefile for stencil benchmark

# CUDA toolkit path - adjust if needed
CUDA_PATH ?= /usr/lib/nvidia-cuda-toolkit

# Compiler and flags
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS = -O3 -arch=sm_70 -lineinfo

# Include directories
INCLUDES = -I$(CUDA_PATH)/include

# Library directories and libraries
LIBDIRS = -L$(CUDA_PATH)/lib64
LIBS = -lcudart -lcurand

# Target executable
TARGET = stencil_benchmark

# Source files
SRC = stencil.cu

# Object files
OBJ = $(SRC:.cu=.o)

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(LIBDIRS) $(LIBS) -o $@ $^

# Compile source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJ) $(TARGET)

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Check CUDA installation
check:
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "NVCC Version:"
	@$(NVCC) --version

.PHONY: all clean run check
