## Comparing basic Matrix operations on CUDA and ROCM
Following table comapares different GPU operations on NVIDIA GPU with CUDA, NVIDIA GPU with ROCM and AMD GPU with ROCM.
To replicate the benchmarks, follow intructions given in the README file. We ran the experiments for 100 times and posted the mean time taken ( in microseconds) and standard deviation( in microseconds) in the table below.


|        Application and its Runtime (mean/std)          | NVIDIA with CUDA           |       NVIDIA with ROCm     |      AMD with ROCm      |
|--------------------------------------------------------|----------------------------|----------------------------|-------------------------|
| Vector Addition (Size=1000000)                         |        39 (11)             |        40 (9)              |      2243 (356)         |
| Matrix Multiplication w/out shared memory   Size=10000 |        51 (9)              |        44 (3)              |      616 (46)           |
| Matrix Multiplication with shared memory    Size=10000 |        15 (2)              |        13 (2)              |       12 (3)            |
  

The GPUS used for experimentation are :
1. NVIDIA GTX TITAN with 12 GB memory.
2. AMD MI60 GPUs

As we can infer from the table that NVIDIA with ROCM takes less time than NVIDIA with CUDA, whem it comes to matrix multiplication. 

## Pytorch GPU Benchmark 
Each network is fed with 12 images with 224x224x3 dimensions. For training, time durations of 20 passes of forward and backward are averaged. For inference, time durations of 20 passes of forward are averaged. 5 warm up steps are performed that do not calculate towards the final result.

### Pytorch Compile On CUDA with NVIDIA GPU
```
Number of GPUs on current device : 1
CUDA Version : 10.0
Cudnn Version : None
Device Name : GeForce GTX TITAN X
mnasnet0_5 model average train time : 105.04903316497803ms
Benchmarking Training float precision type mnasnet0_75
mnasnet0_75 model average train time : 103.99274826049805ms
Benchmarking Training float precision type mnasnet1_0
mnasnet1_0 model average train time : 103.95143032073975ms
Benchmarking Training float precision type mnasnet1_3
mnasnet1_3 model average train time : 113.90717029571533ms
Benchmarking Training float precision type resnet18
resnet18 model average train time : 110.25830268859863ms
Benchmarking Training float precision type resnet34
resnet34 model average train time : 211.5184783935547ms
Benchmarking Training float precision type resnet50
resnet50 model average train time : 258.4140968322754ms
Benchmarking Training float precision type resnet101
resnet101 model average train time : 439.5346403121948ms
Benchmarking Training float precision type resnet152
resnet152 model average train time : 630.8797645568848ms
Benchmarking Training float precision type resnext50_32x4d
resnext50_32x4d model average train time : 861.3776540756226ms
Benchmarking Training float precision type resnext101_32x8d
resnext101_32x8d model average train time : 2011.4038801193237ms
Benchmarking Training float precision type wide_resnet50_2
wide_resnet50_2 model average train time : 463.5503625869751ms
Benchmarking Training float precision type wide_resnet101_2
wide_resnet101_2 model average train time : 833.6229944229126ms
Benchmarking Training float precision type densenet121
densenet121 model average train time : 389.4740152359009ms
Benchmarking Training float precision type densenet169
densenet169 model average train time : 463.42860221862793ms
Benchmarking Training float precision type densenet201
densenet201 model average train time : 567.878098487854ms
Benchmarking Training float precision type densenet161
densenet161 model average train time : 695.3049182891846ms
Benchmarking Training float precision type squeezenet1_0
squeezenet1_0 model average train time : 79.46806907653809ms
Benchmarking Training float precision type squeezenet1_1
squeezenet1_1 model average train time : 53.5618782043457ms
Benchmarking Training float precision type vgg11
vgg11 model average train time : 245.89030265808105ms
Benchmarking Training float precision type vgg11_bn
vgg11_bn model average train time : 273.19058418273926ms
Benchmarking Training float precision type vgg13
vgg13 model average train time : 467.2686529159546ms
Benchmarking Training float precision type vgg13_bn
vgg13_bn model average train time : 492.9059648513794ms
Benchmarking Training float precision type vgg16
vgg16 model average train time : 581.9611549377441ms
Benchmarking Training float precision type vgg16_bn
vgg16_bn model average train time : 609.453558921814ms
Benchmarking Training float precision type vgg19_bn
vgg19_bn model average train time : 723.493332862854ms
Benchmarking Training float precision type vgg19
vgg19 model average train time : 695.1883411407471ms
Benchmarking Training float precision type mobilenet_v2
mobilenet_v2 model average train time : 104.34467315673828ms
```


