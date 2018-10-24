#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#define BLOCK_SIZE 4
 
__global__ void matrixGPU(int *pMatrizA,int *pMatrizB, int *pMatrizResultante, int pColumnasMatrizA, int pFilasMatrizB, int pColumnasMatrizB)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < pColumnasMatrizB && row < pColumnasMatrizA) 
    {
        for(int i = 0; i < pFilasMatrizB; i++) 
        {
            sum += pMatrizA[row * pFilasMatrizB + i] * pMatrizB[i * pColumnasMatrizB + col];
        }
        pMatrizResultante[row * pColumnasMatrizB + col] = sum;
    }
} 

void matrixCPU(int *pMatrizA, int *pMatrizB, int *pMatrizResultante, int pColumnasMatrizA, int pFilasMatrizB, int pColumnasMatrizB) {
    for (int i = 0; i < pColumnasMatrizA; ++i) 
    {
        for (int j = 0; j < pColumnasMatrizB; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < pFilasMatrizB; ++h) 
            {
                tmp += pMatrizA[i * pFilasMatrizB + h] * pMatrizB[h * pColumnasMatrizB + j];
            }
            pMatrizResultante[i * pColumnasMatrizB + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int columnasMatrizA = 4, filasMatrizB = 4, columnasMatrizB = 4; 
    /* Fixed seed for illustration */
    srand(123456987);

    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*columnasMatrizA*filasMatrizB);
    cudaMallocHost((void **) &h_b, sizeof(int)*filasMatrizB*columnasMatrizB);
    cudaMallocHost((void **) &h_c, sizeof(int)*columnasMatrizA*columnasMatrizB);
    cudaMallocHost((void **) &h_cc, sizeof(int)*columnasMatrizA*columnasMatrizB);

    // Rellenando Matriz A y Matriz B
    for (int i = 0; i < columnasMatrizA; ++i) {
        for (int j = 0; j < filasMatrizB; ++j) {
            h_a[i * n + j] = rand() % 1024;
            h_b[i * k + j] = rand() % 1024;
        }
    }

    float tiempoGPU, tiempoCPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inicializa tiempo para GPU
    cudaEventRecord(start, 0);

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*columnasMatrizA*filasMatrizB);
    cudaMalloc((void **) &d_b, sizeof(int)*filasMatrizB*columnasMatrizB);
    cudaMalloc((void **) &d_c, sizeof(int)*columnasMatrizA*columnasMatrizB);

    cudaMemcpy(d_a, h_a, sizeof(int)*columnasMatrizA*filasMatrizB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*filasMatrizB*columnasMatrizB, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (columnasMatrizA + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (columnasMatrizB + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    matrixGPU<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, columnasMatrizA, filasMatrizB, columnasMatrizB);    


    cudaMemcpy(h_c, d_c, sizeof(int)*columnasMatrizA*columnasMatrizB, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // calcula tiempo de ejecucion del GPU
    cudaEventElapsedTime(&tiempoGPU, start, stop);
    printf("GPU time: %f ms.\n\n", tiempoGPU);

    // Inicializa tiempo de ejecucion del CPU
    cudaEventRecord(start, 0);

    matrixCPU(h_a, h_b, h_cc, columnasMatrizA, filasMatrizB,columnasMatrizB);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiempoCPU, start, stop);
    printf("CPU time: %f ms.\n\n", tiempoCPU);

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
