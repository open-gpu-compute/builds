## Comparing basic Matrix operations on CUDA and ROCM
Following table comapares different GPU operations on NVIDIA GPU with CUDA, NVIDIA GPU with ROCM and AMD GPU with ROCM.
To replicate the benchmarks, follow intructions given in the README file. We ran the experiments for 100 times and posted the mean time taken ( in microseconds) and standard deviation( in microseconds) in the table below.
|                                                        |      NVIDIA with CUDA      |       NVIDIA with ROCm     |      AMD with ROCm      |
|--------------------------------------------------------|:----------------:|---------|:----------------:|---------|:-------------:|---------|
|                                                        |     Mean Time    | St. Dev |     Mean Time    | St. Dev | Mean Time     | St. Dev |
| Vector Addition (Size=1000000)                         |        39        |    11   |        40        |    9    |      2243     |   356   |
| Matrix Multiplication w/out shared memory   Size=10000 |        51        |    9    |        44        |    3    |      616      |   46    |
| Matrix Multiplication with shared memory    Size=10000 |        15        |    2    |        13        |    2    |       12      |   3     |


The GPUS used for experimentation are :
1. NVIDIA GTX TITAN with 12 GB memory.
2. AMD MI60 GPUs

As we can infer from the table that NVIDIA with ROCM takes less time than NVIDIA with ROCm, whem it comes to matrix multiplication. 



