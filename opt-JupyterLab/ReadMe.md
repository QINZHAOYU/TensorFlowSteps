# 部署交互式 Python 开发环境 

## JupyterLab 环境

如果你既希望获得本地或云端强大的计算能力，又希望获得 Jupyter Notebook 或 Colab 中方便的在线 Python 交互式运行环境，可以自己为的本地服务器或云服务器安装 JupyterLab。JupyterLab 可以理解成升级版的 Jupyter Notebook/Colab，提供多标签页支持，在线终端和文件管理等一系列方便的功能，接近于一个在线的 Python IDE。

在已经部署 Python 环境后，使用以下命令安装 JupyterLab：

    >>>pip install jupyterlab

然后使用以下命令运行 JupyterLab：

    >>>jupyter lab --ip=0.0.0.0

然后根据输出的提示，使用浏览器访问 `http://服务器地址:8888` ，并使用输出中提供的 token 直接登录（或设置密码后登录）即可。

可以使用 --port 参数指定端口号。

如果需要在终端退出后仍然持续运行 JupyterLab，可以使用 `nohup` 命令及 & 放入后台运行，即：

    >>>nohup jupyter lab --ip=0.0.0.0 &  

程序输出可以在当前目录下的 `nohup.txt` 找到。

为了在 JupyterLab 的 Notebook 中使用自己的 Conda 环境，需要使用以下命令：

    conda activate 环境名
    conda install ipykernel
    ipython kernel install --name 环境名 --user

然后重新启动 JupyterLab，即可在 Kernel 选项和启动器中建立 Notebook 的选项中找到自己的 Conda 环境。


## Jupyter Notebook 环境

*以下安装过程同样适用于 jupyterlab*

首先，创建 conda 环境并激活：

```
conda create -n jupyter python=3.9
source activate jupyter
conda install ipykernel    # 安装内核
```

然后，安装 `jupyterlab` 以及 `jupyter_server`，同时设定 notebook 登录密码 `login password` 并记录对应的密钥 `hash password`：
```
pip install jupyterlab
pip install jupyter_server

python
>>>from jupyter_server.auth import passwd
>>>passwd()
password: <jupyternotebook login password， such as '123abc'>
verify password:
'<generated hash password, used in config file>'
>>>quit()
```

接着，配置 jupyter notebook：

```
jupyter notebook --generate-config              # 生成配置文件并输出文件位置
                                                # 加入--allow-root，解决 root 权限问题

vim /root/.jupyter/jupyter_notebook_config.py   # 修改配置（/string -> Enter， 搜索定位）
'''
c.NotebookApp.password= u'<hash password>'      # 取消注释后顶格写，下同
c.NotebookApp.allow_root=True
...                                             # 根据需要，可以修改ip, port， notebook_dir等配置
'''
```

然后，服务器开启jupyter notebook 服务：

```
jupyter notebook --no-browser            # --no-browser， 由于服务器未安装浏览器，所以在不弹出浏览器的情况下启动

或

nohup jupyter notebook --no-browser &    # nohup，在终端退出后仍然持续运行
```

接着，将服务器端口映射到本地：

```
ssh -N -f -L localhost:8082:localhost:8888 root@121.43.53.155  # 将远程服务器(ip: 121.43.53.155)的8888端口映射到本地的8082端口

<server login password>
```

最后，通过本地浏览器登录服务器中的jupyter notebook：

```
localhost:8082

<login password>
```

如果后期需要在环境中添加库，比如 tensorflow，并加载到jupyter notebook内核:

```
pip install tensorflow

python -m ipykernel install --user --name tensorflow --display-name tf2   # 将 tensorflow 加载到内核并重命名为 tf2
```

不过目前尚有一个问题，就是不同 conda 环境下生成 jupyter notebook 配置文件在同一个位置，这就会导致不同环境下的配置会相互覆盖。
