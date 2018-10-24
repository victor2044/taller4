NVCC = nvcc

all: matrix

%.o : %.cu
	$(NVCC) -c $< -o $@

vecadd : vecadd.o
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a matrix
