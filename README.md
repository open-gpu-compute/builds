## Open GC Dockerfiles

1. Build the Docker image using the command `docker build -t opengc .`
2. Run the container using the command 
```
docker run -it -v $(pwd)/src:/data/src --privileged --rm  --group-add video --gpus all opengc
```
3. Compile `src/vector_add.cu` inside the docker container using the following command:
```
# cd /data/src
# nvcc vector_add.cu -o  vector_add
# ./vector_add
Time taken for addition : 30 microseconds

---------------------------
__SUCCESS__
---------------------------
N                 = 1048576
Threads Per Block = 256
Blocks In Grid    = 4096
---------------------------
```

