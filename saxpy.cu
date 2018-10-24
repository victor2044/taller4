#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#define BLOCK_SIZE 4

__global__ void saxpyGPU(int *pVectorA,int *pVectorB, int *pVectorResultante, int pDimension, int pConstante)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        pVectorResultante[i] = pConstante* pVectorA[i] + pVectorB[i];
    }
} 

void saxpyCPU(int *pVectorA, int *pVectorB, int *pVectorResultante, int pDimension,int pConstante) {
    for (int i = 0; i < pDimension; ++i) 
    {
        int tmp = 0.0;
        tmp += pMatrizA[i * pDimension] * pConstante + pMatrizB[i * pDimension];
        pVectorResultante[i * pDimension] = tmp;
    }
}

int main(int argc, char const *argv[])
{
    int columnasMatrizA = 4, filasMatrizB = 4, columnasMatrizB = 4; 
    int dimension = 4, constante = 8;
    /* Fixed seed for illustration */
    srand(123456987);

    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void *) &h_a, sizeof(int)*dimension);
    cudaMallocHost((void *) &h_b, sizeof(int)*dimension);
    cudaMallocHost((void *) &h_c, sizeof(int)*dimension);
    cudaMallocHost((void *) &h_cc, sizeof(int)*dimension);

    // Rellenando vector A y vector B
    for (int i = 0; i < dimension; ++i) {
            h_a[i * dimension] = rand() % 1024;
            h_b[i * dimension] = rand() % 1024;
    }

    float tiempoGPU, tiempoCPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inicializa tiempo para GPU
    cudaEventRecord(start, 0);

    int *d_a, *d_b, *d_c;
    cudaMalloc((void *) &d_a, sizeof(int)*dimension);
    cudaMalloc((void *) &d_b, sizeof(int)*dimension);
    cudaMalloc((void *) &d_c, sizeof(int)*dimension);

    cudaMemcpy(d_a, h_a, sizeof(int)*dimension, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*dimension, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (dimension + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (dimension + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    multiplicacionGPU<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, dimension,constante);    


    cudaMemcpy(h_c, d_c, sizeof(int)*dimension, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // calcula tiempo de ejecucion del GPU
    cudaEventElapsedTime(&tiempoGPU, start, stop);
    printf("Tiempo de ejecucion GPU: %f ms.\n\n", tiempoGPU);

    // Inicializa tiempo de ejecucion del CPU
    cudaEventRecord(start, 0);

    multiplicacionCPU(h_a, h_b, h_cc, dimension, constante);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiempoCPU, start, stop);
    printf("Tiempo de ejecucion CPU: %f ms.\n\n", tiempoCPU);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}