## Open GC Dockerfiles


0. Clone the repository
```
git clone https://github.com/open-gpu-compute/builds.git
```
1. Build the Docker image using the command
```
docker build -t opengc/opengc .
```
2. Run the container using the command 
```
docker run -it --privileged -v $(pwd)/src:/data/src --rm opengc/opengc
```
4. Hipify `src/vector_add.cu` and compile it using the following command
```
# export HIP_PLATFORM=nvcc # for nvidia gpus
# export HIP_PLATFORM=hcc # for AMD gpus
# cd /data/src
# hipify-perl vector_add.cu > vector_add.hip.cu
# hipcc vector_add.hip.cu -o vector_add.hip.cu.out
# ./vector_add.hip.cu.out
Time taken for addition : 1108 microseconds

---------------------------
__SUCCESS__
---------------------------
N                 = 1048576
Threads Per Block = 256
Blocks In Grid    = 4096
---------------------------
```
4. (Optional, For nvidia GPUS) Compile `src/vector_add.cu` inside the docker container using the following command:
```
# cd /data/src
# nvcc vector_add.cu -o  vector_add.cu.out
# ./vector_add.cu.out
Time taken for addition : 30 microseconds

---------------------------
__SUCCESS__
---------------------------
N                 = 1048576
Threads Per Block = 256
Blocks In Grid    = 4096
---------------------------
```

## Installing Pytorch on AMD 

0. Clone the repository
```
git clone https://github.com/open-gpu-compute/builds.git
```
1. Build the image using Dockerfile. This image  install all the dependencies. 
```
docker build -t opengc .
```
you can also pull the docker image using the command 
```
docker pull opengc/opengc
```
2. Run the container using the command 
```
docker run -it -v $(pwd):/data --privileged --rm  --group-add video  opengc

```
3. Inside the docker run the following command 
```
git clone --recursive https://github.com/ROCmSoftwarePlatform/pytorch.git
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```
4. Hipify the repository
```
python tools/amd_build/build_amd.py
```
5. Run the command to install torch
```
RCCL_DIR=/opt/rocm/rccl/lib/cmake/rccl/ hip_DIR=/opt/rocm/hip/cmake/  BUILD_CAFFE2_OPS=0 PATH=/usr/lib/ccache/:$PATH USE_CUDA=OFF python3 setup.py install
```


## To Run GPU Pytorch Benchmark

0. Clone the repository
```
git clone https://github.com/open-gpu-compute/builds.git
```
1. Build the image Dockerfile.pytorch . This image install torch, torchvision and  all the dependencies. 
```
docker build -t opengc_benchmark_cuda .
```
2. Run the docker using the command 
```
docker run -it -v $(pwd):/data --privileged --rm  --group-add video --shm-size=2gb  --ipc=host --gpus all opengc_benchmark_cuda
```
3. Clone the Gpu benchmark repository inside the docker container.
```
git clone https://github.com/ryujaehun/pytorch-gpu-benchmark.git
```
4. Run the test
```
./test.sh
```
