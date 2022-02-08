# 使用 Docker 部署 TensorFlow 环境

Docker 是轻量级的容器（Container）环境，通过将程序放在虚拟的 “容器” 或者说 “保护层” 中运行，既避免了配置各种库、依赖和环境变量的麻烦，又克服了虚拟机资源占用多、启动慢的缺点。使用 Docker 部署 TensorFlow 的步骤如下：

1. 安装 Docker 。Windows 下，下载官方网站的安装包进行安装即可。Linux 下建议使用 官方的快速脚本 进行安装，即命令行下输入：

```
wget -qO- https://get.docker.com/ | sh
```

如果当前的用户非 root 用户，可以执行 `sudo usermod -aG docker your-user` 命令将当前用户加入 `docker` 用户组。重新登录后即可直接运行 Docker。

Linux 下通过以下命令启动 Docker 服务：

    sudo service docker start

2. 拉取 TensorFlow 映像。Docker 将应用程序及其依赖打包在映像文件中，通过映像文件生成容器。使用 `docker image pull` 命令拉取适合自己需求的 TensorFlow 映像，例如：

```
docker image pull tensorflow/tensorflow:latest-py3        # 最新稳定版本TensorFlow（Python 3.5，CPU版）
docker image pull tensorflow/tensorflow:latest-gpu-py3    # 最新稳定版本TensorFlow（Python 3.5，GPU版）
```

3. 基于拉取的映像文件，创建并启动 TensorFlow 容器。使用 `docker container run` 命令创建一个新的 TensorFlow 容器并启动。

**CPU 版本的** TensorFlow：

    docker container run -it tensorflow/tensorflow:latest-py3 bash

`docker container run` 命令的部分选项如下：

+ `-it` 让 docker 运行的容器能够在终端进行交互，具体而言：
    + `-i` （ `--interactive` ）：允许与容器内的标准输入 (STDIN) 进行交互。
    + `-t` （ `--tty` ）：在新容器中指定一个伪终端。
+ `--rm` ：当容器中的进程运行完毕后自动删除容器。
+ `tensorflow/tensorflow:latest-py3` ：新容器基于的映像。如果本地不存在指定的映像，会自动从公有仓库下载。
+ `bash` 在容器中运行的命令（进程）。Bash 是大多数 Linux 系统的默认 Shell。

**GPU 版本的** TensorFlow：

`docker container run` 命令中添加 `--runtime=nvidia` 选项，并基于具有 GPU 支持的 TensorFlow Docker 映像启动容器即可，即：

    docker container run -it --runtime=nvidia tensorflow/tensorflow:latest-gpu-py3 bash


**Docker 常用命令**:

+ 映像（image）相关操作：

```
docker image pull [image_name]  # 从仓库中拉取映像[image_name]到本机
docker image ls                 # 列出所有本地映像
docker image rm [image_name]    # 删除名为[image_name]的本地映像


# ------------------------------  制作镜像:
项目根目录：/MyProject

在项目的根目录下，新建一个文本文件.dockerignore (类似.gitignore， 排除路径)：
'''
.git
node_modules
npm-debug.log
'''

项目的根目录下，新建一个文本文件 Dockerfile：
'''
FROM node:8.4        # 该 image 文件继承官方的 node image，冒号表示标签，这里是8.4版本的 node。
COPY . /app          # 将当前目录下的所有文件（除了.dockerignore排除的路径），都拷贝进入 image 文件的/app目录。
WORKDIR /app         # 指定接下来的工作路径为/app。
RUN npm install --registry=https://registry.npm.taobao.org   # 在/app目录下，运行npm install命令安装依赖。注意，安装后所有的依赖，都将打包进入 image 文件。
EXPOSE 3000          # 将容器 3000 端口暴露出来， 允许外部连接这个端口。
'''

使用docker image build命令创建 image 文件：
'''
>>> docker image build -t koa-demo:0.0.1  ./

--- 参数说明：
-t，参数用来指定 image 文件的名字；可以用冒号指定标签，如果不指定，默认的标签就是latest。
./，表示 Dockerfile 文件所在的路径。
'''
```

+ 容器（container）相关操作：

```
docker container run [image_name] [command] # 基于[image_name]映像新建并启动容器，并运行[command]
docker container ls                         # 列出本机正在运行的容器
                                            # （加入--all参数列出所有容器，包括已停止运行的容器）
docker container start [containerID]        # 启动已经生成、已经停止运行的ID为[container_id]的容器              
docker container exec -it [containerID] /bin/bash  # 进入一个正在运行的 ID为[container_id]的容器（若docker run 创建容器的时未使用-it参数，一次进入容器。之后就可以在容器的 Shell 执行命令了）
docker container cp [containID]:[/path/to/file] ./ # 从正在运行的 Docker 容器里面，将文件拷贝到本机(这里是当前目录)               
docker container kill [container_id]        # 终止ID为[container_id]的容器（向容器主进程发送 SIGKILL 信号，立即终止）（但依然占据空间）
docker container stop [containerID]         # 停止ID为[container_id]的容器（向容器主进程 先发送SIGTERM 信号（容器程序自行进行收尾清理工作）、 稍后再发送 SIGKILL 信号）
docker container rm [container_id]          # 删除ID为[container_id]的容器
docker container rm $(docker ps -a -q)      # 删除所有容器


# ------------------------------------------- 根据本地镜像制作容器：

从本地 image 文件生成容器：
'''
>>> docker container run -p 8000:3000 -it koa-demo:0.0.1 /bin/bash

--------------  参数说明：
-p：            容器的 3000 端口映射到本机的 8000 端口。
-it：           容器的 Shell 映射到当前的 Shell，然后你在本机窗口输入的命令，就会传入容器。
koa-demo:0.0.1：image 文件的名字（如果有标签，还需要提供标签，默认是 latest 标签）。
/bin/bash：     容器启动以后，内部第一个执行的命令。这里是启动 Bash，保证用户可以使用 Shell
'''


```




