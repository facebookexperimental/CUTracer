NVCCFLAGS := -arch=all -O3 -g -lineinfo -Wno-deprecated-gpu-targets

all: smem_tail.cu
	nvcc $(NVCCFLAGS) smem_tail.cu -o smem_tail

clean:
	rm -f smem_tail
