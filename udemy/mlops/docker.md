https://medium.com/@techsuneel99/docker-from-beginner-to-expert-a-comprehensive-tutorial-5efec10c82ab


![[Pasted image 20250721164537.png]]

![[Pasted image 20250721164612.png]]

# Why use Docker?

1. **Consistency**: Docker ensures that your application runs the same way in every environment, from development to production.
2. **Isolation**: Containers are isolated from each other and the host system, improving security and reducing conflicts.
3. **Portability**: Docker containers can run on any system that supports Docker, regardless of the underlying infrastructure.
4. **Efficiency**: Containers share the host OS kernel, making them more lightweight than traditional virtual machines.
5. **Scalability**: Docker makes it easy to scale applications up or down quickly.





# Docker architecture

Docker uses a client-server architecture. 
The Docker client communicates with the Docker daemon, which does the heavy lifting of building, running, and distributing Docker containers.
The client and daemon can run on the same system, or you can connect a Docker client to a remote Docker daemon.



. ![[Pasted image 20250721165521.png]]


Key components of Docker architecture:

1. **Docker daemon**: The background service running on the host that manages building, running, and distributing Docker containers.
2. **Docker client**: The command-line tool that allows users to interact with the Docker daemon.
3. **Docker registries**: Repositories that store Docker images. Docker Hub is a public registry that anyone can use.
4. **Docker objects**: Images, containers, networks, volumes, plugins, and other objects.
# Installing Docker

```
sudo apt-get update
```

2. Install packages to allow apt to use a repository over HTTPS:

```
sudo apt-get install \  
    apt-transport-https \  
    ca-certificates \  
    curl \  
    gnupg-agent \  
    software-properties-common
    
```


3. Add Docker’s official GPG key:
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

4. Set up the stable repository:
```
sudo add-apt-repository \  
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \  
   $(lsb_release -cs) \  
   stable"
```

5. Update the package index again and install Docker:
```
sudo apt-get update  
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
5. Verify that Docker is installed correctly:

```
sudo docker run hello-world
```




# CLI

```bash
# Docker CLI
# The Docker Command Line Interface (CLI) is the primary way to interact with Docker.

# 1. Check Docker version
docker version

# 2. Display system-wide information
docker info

# 3. List all available Docker commands
docker

# ------------------------------
# Docker Images
# Docker images are read-only templates used to create containers.
# They include the application code, libraries, dependencies, and configuration files.

# 1. List locally available Docker images
docker images

# 2. Pull the latest Ubuntu image from Docker Hub
docker pull ubuntu:latest
```

#  Running your first container
```
docker run -it ubuntu:latest /bin/bash
```
This command does the following:

- `docker run`: Creates and starts a new container
- `-it`: Provides an interactive terminal
- `ubuntu:latest`: Specifies the image to use
- `/bin/bash`: Command to run inside the container

You should now be inside the Ubuntu container. Try running some commands:

```
ls  
cat /etc/os-release
```
To exit the container, type `exit` or press `Ctrl+D`.

# Container lifecycle

``` shell
# Create a new container without starting it
docker create --name mycontainer ubuntu:latest

# Start a stopped container
docker start mycontainer

# Stop a running container
docker stop mycontainer

# Restart a container (stops and then starts it)
docker restart mycontainer

# Pause a running container, suspending all processes inside it
docker pause mycontainer

# Unpause a paused container, resuming its processes
docker unpause mycontainer

# Remove a stopped container
# Note: To remove a running container, you would use 'docker rm -f mycontainer'
docker rm mycontainer
```


Note: You can’t remove a running container unless you use the `-f` (force) option.

# Listing and inspecting containers

``` shell
# List all currently running containers
docker ps

# List all containers, including those that are stopped or exited
docker ps -a

# Display detailed low-level information about a container in JSON format
docker inspect mycontainer

# Fetch and display the logs of a container
# To view logs in real-time as they are generated, add the -f (follow) flag: docker logs -f mycontainer
docker logs mycontainer
```


Note : Add the `-f` flag to follow the logs in real-time.


# Running containers in detached mode

So far, we’ve run containers in interactive mode. For long-running services, you’ll often want to run containers in detached mode:
``` shell
docker run -d --name mywebserver nginx
```

This starts an Nginx web server container in the background. You can still interact with it using other Docker commands.


## detached mode

Docker's **detached mode** (activated using the `-d` or `--detach` flag with `docker run` or implicitly with `docker start`) allows you to run a container in the background, freeing up your terminal for other tasks.

Here's a brief description:

- **Background Execution:** When you start a container in detached mode, Docker runs it as a background process. This means your command prompt immediately returns, and you can continue to use your terminal for other commands without being "attached" to the container's standard input, output, or error streams.
    
- **Non-Interactive:** Detached containers are ideal for long-running services or applications that don't require direct user interaction or continuous monitoring of their output in the terminal (e.g., web servers, databases, APIs).
    
- **Output Management:** Even though the output isn't displayed directly, you can still view the container's logs using `docker logs <container_name_or_id>` (and even follow them in real-time with `docker logs -f`).
    
- **Management:** You can manage detached containers using other Docker commands like `docker ps` (to list running containers), `docker stop`, `docker restart`, `docker inspect`, and `docker exec` (to run commands inside a running container).
    
- **Container ID:** When you start a container in detached mode, Docker typically prints the container's long ID to your terminal, which you can use for subsequent commands.
    

In essence, detached mode enables Docker containers to operate like background services, allowing for efficient multitasking and automated processes without tying up your terminal.

# Working with Docker Images

``` shell
# --- Finding and Pulling Images ---

# Search for images on Docker Hub (e.g., searching for "nginx" images)
docker search nginx

# Pull an image from Docker Hub to your local machine
# 'latest' is the default tag if not specified. You can specify a version, e.g., nginx:1.19
docker pull nginx:latest


# --- Creating Images ---

# Method 1: Committing changes made in a container (less common, but useful for quick experiments)

# Run an interactive container with the latest Ubuntu image and get a bash shell
docker run -it ubuntu:latest /bin/bash
# Inside the container, update package lists and install nginx
# apt-get update
# apt-get install -y nginx
# Exit the container when done
# exit

# Commit the changes from the container (replace <container_id> with the actual ID)
# This creates a new image from the state of the container
docker commit <container_id> my-nginx-image:v1


# Method 2: Building from a Dockerfile (preferred and standard method)

# Create a file named 'Dockerfile' (no extension) with instructions:
# FROM ubuntu:latest                 # Base image for our new image
# RUN apt-get update && apt-get install -y nginx  # Execute commands to install nginx
# EXPOSE 80                          # Document that the container listens on port 80
# CMD ["nginx", "-g", "daemon off;"] # Default command to run when a container starts from this image

# Build an image from the Dockerfile in the current directory
# The '-t' flag tags the image with a name and optional version (e.g., my-nginx-image:v2)
# The '.' at the end specifies the build context (current directory)
docker build -t my-nginx-image:v2 .


# --- Managing Images ---

# List all local Docker images
docker images

# Remove an image by its name and tag (or ID). You cannot remove an image if a container is using it.
docker rmi my-nginx-image:v1

# Tag an existing local image with a new name and/or tag.
# Useful for preparing an image to be pushed to a registry like Docker Hub.
# (Replace 'my-dockerhub-username' with your actual Docker Hub username)
docker tag my-nginx-image:v2 my-dockerhub-username/my-nginx-image:v2

# Push an image to a configured remote registry (e.g., Docker Hub)
# You need to be logged in to Docker Hub using 'docker login' first.
docker push my-dockerhub-username/my-nginx-image:v2


# --- Image Layers and Caching ---

# Display the history and layers of an image
# This shows each instruction from the Dockerfile as a layer and its size
docker history my-nginx-image:v2
```

#  Creating and Managing Docker Containers

```shell
# --- Running Containers with Various Options ---

# Run a container, print "Hello, World!", and automatically remove it upon exit
docker run --rm ubuntu:latest echo "Hello, World!"

# Run a container with a custom, user-defined name
docker run --name my-custom-container ubuntu:latest

# Run a container and publish a port from the container to the host
# This maps port 8080 on the host to port 80 inside the nginx container
docker run -p 8080:80 nginx

# Run a container and set an environment variable within it
# The 'env' command inside the container will then display this variable
docker run -e MY_VAR=my_value ubuntu:latest env

# Run a container with a limited amount of memory (e.g., 512 megabytes)
docker run --memory=512m ubuntu:latest


# --- Executing Commands in Running Containers ---

# Execute an interactive shell command inside a running container
# '-it' combines -i (interactive) and -t (pseudo-TTY) to provide a proper shell experience
docker exec -it my-custom-container /bin/bash


# --- Copying Files Between Host and Container ---

# Copy a file from the host system to a specific path inside a container
docker cp ./myfile.txt my-custom-container:/path/in/container/

# Copy a file from a specific path inside a container to the host system
docker cp my-custom-container:/path/in/container/myfile.txt ./


# --- Monitoring Containers ---

# View real-time resource usage statistics (CPU, Memory, Network I/O) for running containers
docker stats

# View the processes running inside a specific container
docker top my-custom-container


# --- Container Resource Constraints ---

# Limit the container's CPU usage to a percentage (e.g., 0.5 means 50% of one CPU core)
docker run --cpus=0.5 ubuntu:latest

# Set relative CPU shares for a container. Default is 1024.
# A container with 512 shares gets half the CPU time compared to a default one if under contention.
docker run --cpu-shares=512 ubuntu:latest

# Limit the container's memory usage and enable swapping
# This limits RAM to 1GB and allows up to 2GB of swap space
docker run --memory=1g --memory-swap=2g ubuntu:latest


# --- Updating Containers ---

# Update the resource constraints of a running container (e.g., set CPU to 1 core, memory to 2GB)
docker update --cpus=1 --memory=2g my-custom-container

# Rename an existing container
docker rename my-custom-container my-new-container-name


# --- Container Restart Policies ---

# Run a container with a restart policy that always restarts it if it stops, unless explicitly stopped by the user
docker run --restart=always nginx
# Restart Policies:
#   no: Never restart the container automatically (default).
#   on-failure[:max-retries]: Restart only if the container exits with a non-zero status code (indicating an error).
#   always: Always restart the container, regardless of exit status, unless it was explicitly stopped.
#   unless-stopped: Always restart the container unless it was explicitly stopped by the user.


# --- Attaching to and Detaching from Containers ---

# Attach your terminal's standard input, output, and error streams to a running container
docker attach my-custom-container

# Detach from an attached container without stopping it
# This is done by pressing 'Ctrl-p' followed by 'Ctrl-q'

```

#  Docker Networking

## Docker Networking: Connecting Containers and the World

Docker networking is a fundamental aspect of building robust and scalable applications with containers. It enables containers to communicate with each other, with the Docker host, and with external networks.

## Network Drivers

Docker provides several built-in network drivers, each serving different purposes:

* **`bridge`**:
    * **Description**: This is the **default network driver** when you don't specify one. It creates a private internal network for containers on a single Docker host. Containers on the same bridge network can communicate with each other, and they can also communicate with the outside world via NAT (Network Address Translation).
    * **Use Case**: Ideal for standalone containers or small, multi-container applications running on a single host.

* **`host`**:
    * **Description**: This driver **removes network isolation** between the container and the Docker host. The container directly uses the host's network stack, meaning it shares the host's IP address and port space.
    * **Use Case**: When you need extreme network performance or specific network configurations not easily achievable with bridge networks, and you are comfortable with the container sharing the host's network.

* **`overlay`**:
    * **Description**: Designed for **multi-host container communication** and used primarily in Docker Swarm mode. It creates a distributed network across multiple Docker daemons, allowing containers on different hosts to communicate seamlessly as if they were on the same network.
    * **Use Case**: Orchestrating applications across a cluster of Docker hosts (e.g., in a Swarm).

* **`macvlan`**:
    * **Description**: This driver allows you to **assign a MAC address to a container's network interface**, making the container appear as a physical device on your network. This means the container gets its own IP address from your router/DHCP server on the physical network, rather than being behind NAT.
    * **Use Case**: When you need your containers to be directly addressable on your physical network, or for legacy applications that expect to be on a physical network.

* **`none`**:
    * **Description**: This driver **disables all networking** for a container. The container will not have an IP address and cannot communicate with any other network resources.
    * **Use Case**: For containers that perform specific, isolated tasks and do not require network connectivity.

## Listing and Inspecting Networks

```bash
# List all Docker networks available on the host
docker network ls

# Inspect a specific network (e.g., the default 'bridge' network)
# This provides detailed JSON output about the network's configuration, connected containers, etc.
docker network inspect bridge
```


## Creating Custom Networks

It's highly recommended to create custom bridge networks for your applications instead of relying on the default bridge network. This provides better isolation and easier service discovery.

Bash

```shell
# Create a custom bridge network named 'my-custom-network'
docker network create --driver bridge my-custom-network
```

## Connecting Containers to Networks

Containers can be connected to multiple networks, allowing for flexible communication patterns.

Bash

```shell
# Connect a running container (my-container) to a custom network (my-custom-network)
docker network connect my-custom-network my-container

# Disconnect a running container (my-container) from a custom network (my-custom-network)
docker network disconnect my-custom-network my-container

# Run a new container (nginx) and connect it directly to a specified network
docker run --network my-custom-network nginx
```

## Container DNS (Service Discovery)

A significant advantage of user-defined networks is built-in service discovery. Containers on the same user-defined network can resolve each other by their container names.

Bash

```shell
# 1. Create a custom network for your application
docker network create my-app-network

# 2. Run the 'web' container (nginx) on this network
docker run -d --name web --network my-app-network nginx

# 3. Run the 'db' container (postgres) on the same network
docker run -d --name db --network my-app-network postgres

# Now, from inside the 'web' container, you can access the 'db' container
# using 'db' as the hostname (e.g., ping db or connect to 'db' at port 5432)
# Example: docker exec -it web ping db
```

## Port Mapping (Publishing Ports)

To make a container's internal port accessible from the Docker host or external networks, you need to "publish" or "map" it.

Bash

```shell
# Publish port 80 (inside the nginx container) to port 8080 on the host
# Traffic to host:8080 will be routed to container:80
docker run -p 8080:80 nginx

# Publish port 80 (inside the nginx container) to port 8080 on the host,
# specifically binding it to the host's loopback IP address (127.0.0.1)
# This makes the service accessible only from the host machine itself.
docker run -p 127.0.0.1:8080:80 nginx
```

## Network Troubleshooting

When facing network issues with containers, these commands are invaluable for diagnosis:

Bash

```shell
# 1. Check a container's detailed network settings in JSON format
docker inspect --format '{{json .NetworkSettings.Networks}}' my-container

# 2. Use 'docker exec' to run network diagnostic tools directly inside the container
# Ping an external host to check outbound connectivity
docker exec -it my-container ping google.com
# List open ports and network connections inside the container
docker exec -it my-container netstat -tuln

# 3. Review container logs for any network-related errors or connection failures
docker logs my-container

# 4. If you suspect DNS resolution issues, examine the container's DNS configuration
docker exec -it my-container cat /etc/resolv.conf
```

## Advanced Networking Topics

- **Exposing Containers (`--expose`)**:
    
    - **Description**: The `--expose` flag (used with `docker run`) informs Docker that the container listens on the specified ports at runtime. This **does not publish the port** to the host machine. It primarily serves as documentation and can be used by other Docker services (like Swarm) for inter-container communication.
        
    - **Use Case**: When you want to indicate which ports your service uses, especially for multi-service applications where containers communicate internally without needing external access.
        
    
    Bash
    
    ```shell
    docker run --expose 80 nginx
    ```
    
- **Link Containers (`--link`) - (Legacy Feature)**:
    
    - **Description**: The `--link` flag was an older method for enabling communication between containers on the default bridge network. It injected environment variables (like IP addresses and port information) into the linked container. **It is now considered a legacy feature**, and **user-defined networks with their built-in DNS are the preferred and more robust solution**.
        
    - **Use Case**: Mostly for understanding older Docker setups. **Avoid using it for new deployments.**
        
    
    Bash
    
    ```shell
    # (Deprecated) Run a 'db' container
    docker run --name db postgres
    # (Deprecated) Run an 'nginx' container and link it to 'db' as 'database'
    docker run --link db:database nginx
    ```
    
- **MacVLAN Networks**:
    
    - **Description**: MacVLAN networks allow containers to have their own unique MAC addresses and IP addresses on the physical network. They bypass the Docker bridge and NAT, making the container appear as a first-class citizen on your network segment.
        
    - **Use Case**: Scenarios where containers need to be directly accessible from the physical LAN, or for integrating with existing network devices that expect MAC-based addressing. Requires a specific network interface (e.g., `eth0`) on the host.
        
    
    Bash
    
    ``` shell
    # Create a macvlan network, specifying the subnet, gateway, and parent interface (e.g., eth0)
    docker network create -d macvlan \
      --subnet=192.168.1.0/24 \
      --gateway=192.168.1.1 \
      -o parent=eth0 my-macvlan-net
    ```


# Writing Dockerfiles

A Dockerfile is a simple text file that contains a series of instructions. Docker reads these instructions sequentially to build a Docker image. Dockerfiles ensure your image builds are reproducible, transparent, and version-controlled.

## Dockerfile Basics

Here's a breakdown of a typical Dockerfile and its core instructions:

``` dockerfile

FROM ubuntu:20.04                                   # Specifies the base image for this build. All subsequent instructions run on top of this.
RUN apt-get update && apt-get install -y nginx     # Executes commands in a new layer to install software within the image.
COPY ./my-nginx.conf /etc/nginx/nginx.conf         # Copies files from the local build context (host) into the image.
EXPOSE 80                                          # Informs Docker that the container will listen on port 80 at runtime (documentation, not publishing).
CMD ["nginx", "-g", "daemon off;"]                 # Provides the default command to execute when a container starts from this image.
```

### Common Dockerfile Instructions Explained:

- **`FROM <image>[:<tag>]`**:
    
    - **Purpose**: Specifies the base image upon which your new image will be built. Every Dockerfile must start with a `FROM` instruction (or `ARG` followed by `FROM`).
        
    - **Example**: `FROM ubuntu:20.04`
        
- **`RUN <command>`**:
    
    - **Purpose**: Executes commands in a new layer on top of the current image. The result of each `RUN` instruction is committed as a new image layer.
        
    - **Example**: `RUN apt-get update && apt-get install -y nginx`
        
- **`COPY <source> <destination>`**:
    
    - **Purpose**: Copies new files or directories from the host's build context into the filesystem of the image at the specified path.
        
    - **Example**: `COPY ./my-nginx.conf /etc/nginx/nginx.conf`
        
- **`ADD <source> <destination>`**:
    
    - **Purpose**: Similar to `COPY`, but with additional capabilities. It can handle remote URLs (downloading files) and automatically extract compressed files (tar, gzip, bzip2, etc.) if the destination is a directory.
        
    - **Example**: `ADD http://example.com/file.tar.gz /tmp/`
        
- **`EXPOSE <port> [<port>...]`**:
    
    - **Purpose**: Informs Docker that the container will listen on the specified network ports at runtime. This is purely declarative/documentation; it does not actually publish the port to the host. For publishing, use `-p` with `docker run`.
        
    - **Example**: `EXPOSE 80 443`
        
- **`CMD ["executable", "param1", "param2"]` (exec form)** or **`CMD command param1 param2` (shell form)**:
    
    - **Purpose**: Provides defaults for an executing container. There can only be one `CMD` instruction in a Dockerfile. If multiple are listed, only the last one takes effect. If `docker run` specifies a command, the `CMD` instruction is ignored.
        
    - **Example**: `CMD ["nginx", "-g", "daemon off;"]`
        
- **`ENTRYPOINT ["executable", "param1", "param2"]` (exec form)** or **`ENTRYPOINT command param1 param2` (shell form)**:
    
    - **Purpose**: Configures a container to run as an executable. When an `ENTRYPOINT` is defined, the `CMD` (if present) becomes arguments to the `ENTRYPOINT`. `docker run` arguments are also appended to the `ENTRYPOINT`.
        
    - **Example**: `ENTRYPOINT ["/usr/bin/my-app"]` (with `CMD ["--debug"]`, `my-app --debug` would run)
        

## Best Practices for Writing Dockerfiles

Following these practices leads to more efficient, smaller, and secure images:

1. **Use Official Base Images**:
    
    - **Description**: Leverage images from trusted sources like Docker Hub's official repositories (e.g., `ubuntu`, `nginx`, `node`). They are usually well-maintained, secure, and optimized.
        
2. **Minimize the Number of Layers**:
    
    - **Description**: Each `RUN`, `COPY`, `ADD` instruction creates a new layer. Combining multiple commands into a single `RUN` instruction significantly reduces the number of layers, leading to smaller image sizes and faster builds (due to less caching overhead). Use `&& \` for line continuation.
        
    - **Example**:
        
        Dockerfile
        
        ``` dockerfile
        RUN apt-get update && \
            apt-get install -y \
            package1 \
            package2 \
            package3 && \
            rm -rf /var/lib/apt/lists/* # Clean up apt cache to further reduce image size
        ```
        
3. **Use `.dockerignore`**:
    
    - **Description**: Similar to `.gitignore`, a `.dockerignore` file in your build context specifies files and directories to exclude from being sent to the Docker daemon during the build process. This speeds up builds and prevents sensitive information or unnecessary large files from being included.
        
    - **Example**:
        
        ```shell 
        .git
        node_modules
        tmp/
        *.log
        ```
        
4. **Use Multi-Stage Builds**:
    
    - **Description**: This is a powerful feature that allows you to use multiple `FROM` statements in a single Dockerfile. Each `FROM` starts a new build stage. You can copy artifacts from a previous stage to a later stage, leaving behind all the build-time dependencies, compilers, and development tools that are not needed in the final runtime image.
        
    - **Benefit**: Significantly reduces the final image size and attack surface.
        
    - **Example**: (See detailed example in "Multi-stage Builds" section below)
        
5. **Set the `WORKDIR`**:
    
    - **Description**: Defines the working directory for any `RUN`, `CMD`, `ENTRYPOINT`, `COPY`, and `ADD` instructions that follow it in the Dockerfile. It helps organize your image's filesystem.
        
    - **Example**:
        
        Dockerfile
        
        ```dockerfile
        WORKDIR /app
        COPY . .
        ```
        
6. **Use Environment Variables (`ENV`)**:
    
    - **Description**: Sets environment variables within the image. These variables persist when a container is run from the image and can be accessed by applications inside the container.
        
    - **Use Case**: Configuring application settings, paths, or default values.
        
    - **Example**:
        
        Dockerfile
        
        ```dockerfile
        ENV APP_HOME /app
        WORKDIR $APP_HOME
        ```
        
7. **Use Labels (`LABEL`)**:
    
    - **Description**: Adds metadata to an image as key-value pairs. This metadata can include information like maintainer, version, description, license, etc. It's useful for organizing and querying images.
        
    - **Example**:
        
        Dockerfile
        
        ```dockerfile
        LABEL maintainer="your-email@example.com" \
              version="1.0" \
              description="This is my custom image for a web application"
        ```
        

## Building Images from Dockerfiles

To build a Docker image from a Dockerfile, navigate to the directory containing your `Dockerfile` (and any files it `COPY`s or `ADD`s) and run:

Bash

```shell
# Build an image with the tag 'my-image' and version 'v1' from the Dockerfile in the current directory.
# The '.' at the end specifies the build context (the directory Docker will use for COPY/ADD operations).
docker build -t my-image:v1 .
```

## Dockerfile Instructions in Depth

Some instructions offer more advanced functionality:

- **`HEALTHCHECK [OPTIONS] CMD command`**:
    
    - **Purpose**: Tells Docker how to test if a container is still working. This is crucial for orchestrators (like Docker Swarm, Kubernetes) to determine if a service instance is healthy and should remain running.
        
    - **Options**:
        
        - `--interval=<duration>`: How often to run the check (default: 30s)
            
        - `--timeout=<duration>`: How long to wait for a command to complete (default: 30s)
            
        - `--start-period=<duration>`: Initialization time to allow a container to start up before applying health checks (default: 0s)
            
        - `--retries=<number>`: How many consecutive failures until the container is considered unhealthy (default: 3)
            
    - **Example**:
        
        Dockerfile
        
        ```dockerfile
        HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
          CMD curl -f http://localhost/ || exit 1
        ```
        
- **`ARG <name>[=<default value>]`**:
    
    - **Purpose**: Defines variables that users can pass to the builder at build-time using the `--build-arg` flag. Unlike `ENV`, `ARG` variables are not available in the running container (unless an `ENV` instruction sets an environment variable using an `ARG`'s value).
        
    - **Example**:
        
        Dockerfile
        
        ```dockerfile
        ARG VERSION=latest
        FROM ubuntu:${VERSION}
        ```
        
        Then build with: `docker build --build-arg VERSION=22.04 -t my-app:22.04 .`
        
- **`VOLUME ["/path/to/data"]` or `VOLUME /path/to/data`**:
    
    - **Purpose**: Creates a mount point with the specified name and marks it as holding externally mounted volumes. This indicates that a specified directory or file should be stored outside the container's writable layer, allowing data to persist even if the container is removed or updated.
        
    - **Example**: `VOLUME /app/data`
        
- **`USER <user>[:<group>]` or `USER <UID>[:<GID>]`**:
    
    - **Purpose**: Sets the user name or UID (and optionally the group name or GID) to use when running the image, and for any `RUN`, `CMD`, and `ENTRYPOINT` instructions that follow it. Running as a non-root user is a security best practice.
        
    - **Example**:
        
        Dockerfile
        
        ```dockerfile
        RUN groupadd -r mygroup && useradd -r -g mygroup myuser
        USER myuser
        ```
        

## Multi-Stage Builds

Multi-stage builds are a cornerstone of efficient Docker image creation. They allow you to define multiple temporary build environments (stages) and then selectively copy only the necessary artifacts (executables, configuration files) from a previous stage to a smaller, final runtime image. This eliminates large build-time dependencies (like compilers, SDKs, development tools) from your production images.

### Example of a Multi-Stage Build for a Go Application:

Dockerfile

``` dockerfile
# Stage 1: The 'builder' stage
FROM golang:1.16 AS builder # Use a Go base image for compilation
WORKDIR /app                # Set the working directory inside this stage
COPY . .                    # Copy all source code into the builder stage
RUN go mod download         # Download Go modules
# Compile the Go application. CGO_ENABLED=0 creates a statically linked binary, GOOS=linux targets Linux.
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# Stage 2: The 'final' runtime stage
# Use a minimal base image like Alpine for a small footprint
FROM alpine:3.14
# Install necessary runtime dependencies (e.g., CA certificates for HTTPS connections)
RUN apk --no-cache add ca-certificates
WORKDIR /root/ # Set the working directory for the final image
# Copy only the compiled binary from the 'builder' stage to the final image
COPY --from=builder /app/main .
# Define the command to run when a container starts from this final image
CMD ["./main"]
```

**Benefits of this multi-stage example:**

- **Smaller Final Image**: The `golang:1.16` image is large (hundreds of MBs) as it contains the Go compiler and tools. The `alpine:3.14` image is tiny (a few MBs). By only copying the compiled `main` binary, the final image is significantly smaller.
    
- **Reduced Attack Surface**: Fewer unnecessary tools and libraries in the production image mean fewer potential vulnerabilities.
    
- **Cleaner Images**: The final image contains only what's absolutely essential for the application to run.
    
- **Separation of Concerns**: Clearly separates the build environment from the runtime environment.
    

Understanding Dockerfiles in depth, especially multi-stage builds, is essential for creating optimized and secure Docker images. With this knowledge, you're well-equipped to define complex application environments.