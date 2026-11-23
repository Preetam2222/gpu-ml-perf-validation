CXX ?= g++
NVCC ?= nvcc
all: ceilings_cpu
gpu: ceilings_gpu
ceilings_cpu: ceilings_cpu.cpp
	$(CXX) -O3 -march=native -fopenmp -o $@ $<
ceilings_gpu: ceilings_gpu.cu
	$(NVCC) -O3 -o $@ $<
clean:
	rm -f ceilings_cpu ceilings_gpu
