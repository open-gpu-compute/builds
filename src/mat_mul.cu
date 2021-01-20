/*
Compiling with nvcc:
nvcc mat_mul.cu -o mat_mul -std=c++11
./mat_mul
Sample Output:
[Enter size of square matrix]
100
[matrix multiplication of 100 elements]
Time taken for matrix multiplication without shared memory : 20 microseconds
Time taken for matrix multiplication with shared memory : 9 microseconds
*/

// Matrix multiplication with and without shared memory
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; // 
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Get a matrix element
__device__ float GetElement(const Matrix mat, int row, int col)
{
    return mat.elements[row * mat.stride + col];
}

// Set mat matrix element
__device__ void SetElement(Matrix mat, int row, int col,
                           float value)
{
    mat.elements[row * mat.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix mat, int row, int col) 
{
    Matrix matsub;
    matsub.width    = BLOCK_SIZE;
    matsub.height   = BLOCK_SIZE;
    matsub.stride   = mat.stride;
    matsub.elements = &mat.elements[mat.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return matsub;
}



// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernelSharedMemory(const Matrix, const Matrix, Matrix);
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    auto start = high_resolution_clock::now();// Calculate Execution Time
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken for matrix multiplication without shared memory : "<< duration.count() << " microseconds"<<"\n";
    auto start1 = high_resolution_clock::now();// Calculate Execution Time
    MatMulKernelSharedMemory<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop1 - start1);
    cout << "Time taken for matrix multiplication with shared memory : "<< duration1.count() << " microseconds"<<"\n";


    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// Matrix multiplication kernel using shared memory()
// The following code sample is an implementation of 
// matrix multiplication that does take advantage of 
// shared memory. In this implementation, each thread 
// block is responsible for computing one square sub-matrix
//  Csub of C and each thread within the block is responsible
// for computing one element of Csub.
 __global__ void  MatMulKernelSharedMemory(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

int main(void)
{

    // Print the matrix length to be used, and compute its size
    int matSize;
    printf("[Enter size of square matrix]\n");
    scanf("%d",&matSize);
    Matrix h_A,h_B,h_C;
    h_A.width = h_B.width = h_C.width = matSize;
    h_A.height = h_B.height = h_C.height = matSize;
    size_t size = matSize * matSize * sizeof(float);
    printf("[matrix multiplication of %d elements]\n", matSize);

    // Allocate the host input matrix A
    h_A.elements = (float *)malloc(size);

    // Allocate the host input matrix B
    h_B.elements = (float *)malloc(size);

    // Allocate the host output matrix C
    h_C.elements = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A.elements == NULL || h_B.elements == NULL || h_C.elements == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input matrix
    for (int i = 0; i < matSize; ++i)
        for (int j= 0; j < matSize; ++j)
    {
        {
            h_A.elements[i * matSize + j] = rand()/(float)RAND_MAX;
            h_B.elements[i * matSize + j] = rand()/(float)RAND_MAX;
        }
    } 
    MatMul(h_A,h_B,h_C);
} 
    
