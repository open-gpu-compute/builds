
/*
Compiling with nvcc:
nvcc mat_add.cu -o mat_add -std=c++11
./mat_add
Sample Output:
[Enter size of matrix]
100
[matrix addition of 100 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with dimension (7, 7) blocks of dimension (16, 16) threads
Time taken for addition : 21 microseconds
Copy output data from the CUDA device to the host memory
Done
*/

// Matrix addition using CUDA C++
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;

__global__ void matrixAdd(float **A, float **B, float **C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < numElements && j< numElements)
    {
        C[i][j] = A[i][j] + B[i][j];
    }
}

int main(void)
{

    // Print the matrix length to be used, and compute its size
    int numElements;
    printf("[Enter size of matrix]\n");
    scanf("%d",&numElements);
    
    size_t size = numElements * numElements * sizeof(float);
    printf("[matrix addition of %d elements]\n", numElements);

    // Allocate the host input matrix A
    float *h_A = (float *)malloc(size);

    // Allocate the host input matrix B
    float *h_B = (float *)malloc(size);

    // Allocate the host output matrix C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input matrix
    for (int i = 0; i < numElements; ++i)
        for (int j= 0; j < numElements; ++j)
    {
        {
            h_A[i * numElements + j] = rand()/(float)RAND_MAX;
            h_B[i * numElements + j] = rand()/(float)RAND_MAX;
        }
    }   

    // Allocate the device input matrix A
    float **d_A = NULL;
    cudaMalloc((void **)&d_A, size);

    // Allocate the device input matrix B
    float **d_B = NULL;
    cudaMalloc((void **)&d_B, size);



    // Allocate the device output matrix C
    float **d_C = NULL;
    cudaMalloc((void **)&d_C, size);


    // Copy the host input matrix A and B in host memory to the device input matrix in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

   

    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Specfic number of threads per block and number of 
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numElements -1) / threadsPerBlock.x+1, (numElements-1)/ threadsPerBlock.y+1);

    printf("CUDA kernel launch with dimension (%d, %d) blocks of dimension (%d, %d) threads\n", blocksPerGrid.x,blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
    // Launch the matrix Add CUDA Kernel

    auto start = high_resolution_clock::now();// Calculate Execution Time
    matrixAdd<<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0>>> (d_A, d_B, d_C, numElements);    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken for addition : "<< duration.count() << " microseconds"<<"\n";

    

    // Copy the device result matrix in device memory to the host result matrix
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state.
    cudaDeviceReset();

    printf("Done\n");
    return 0;
}