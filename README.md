## Open GC Dockerfiles

1. Build the Docker image using the command `docker build -t opengc .`
2. Run the container using the command 
```
docker run -it -v $(pwd)/src:/data/src --privileged --rm  --group-add video --gpus all opengc
```
