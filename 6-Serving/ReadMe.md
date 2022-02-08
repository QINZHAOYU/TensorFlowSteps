# 模型的服务器部署

当我们将模型训练完毕后，往往需要将模型在生产环境中部署。最常见的方式，是在服务器上提供一个 API，即客户机向服务器的某个 API 发送特定格式的请求，服务器收到请求数据后通过模型进行计算，并返回结果。  
如果仅仅是做一个 Demo，不考虑高并发和性能问题，其实配合 Flask 等 Python 下的 Web 框架就能非常轻松地实现服务器 API。  
不过，如果是在真的实际生产环境中部署，这样的方式就显得力不从心了。这时，TensorFlow 为我们提供了 TensorFlow Serving 这一组件，能够帮助我们在实际生产环境中灵活且高性能地部署机器学习模型。


## TensorFlow Serving 安装

TensorFlow Serving 可以使用 apt-get 或 Docker 安装。在生产环境中，推荐 使用 [Docker 部署 TensorFlow Serving](https://www.tensorflow.org/tfx/serving/docker) 。

首先设置安装源：

    # 添加Google的TensorFlow Serving源
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
    # 添加gpg key
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

更新源后，即可使用 apt-get 安装 TensorFlow Serving

    sudo apt-get update
    sudo apt-get install tensorflow-model-server

在运行 curl 和 apt-get 命令时，可能需要设置代理。

curl 设置代理的方式为 `-x` 选项或设置 `http_proxy` 环境变量，即

    export http_proxy=http://代理服务器IP:端口
    或
    curl -x http://代理服务器IP:端口 URL

apt-get 设置代理的方式为 -o 选项，即

    sudo apt-get -o Acquire::http::proxy="http://代理服务器IP:端口" ...


## TensorFlow Serving 模型部署

TensorFlow Serving 可以直接读取 SavedModel 格式的模型进行部署，使用以下命令即可：

    tensorflow_model_server \
        --rest_api_port=端口号（如8501） \
        --model_name=模型名 \
        --model_base_path="SavedModel格式模型的文件夹绝对地址（不含版本号）"

TensorFlow Serving 支持热更新模型，其典型的模型文件夹结构如下：

    /saved_model_files
        /1      # 版本号为1的模型文件
            /assets
            /variables
            saved_model.pb
        ...
        /N      # 版本号为N的模型文件
            /assets
            /variables
            saved_model.pb

上面 1~N 的子文件夹代表不同版本号的模型。当指定 `--model_base_path` 时，只需要指定根目录的 绝对地址 （不是相对地址）即可。例如，如果上述文件夹结构存放在 `home/snowkylin` 文件夹内，则 `--model_base_path` 应当设置为 `home/snowkylin/saved_model_files` （不附带模型版本号）。    **TensorFlow Serving 会自动选择版本号最大的模型进行载入**。

如果想要多个版本的模型同时存在，且多个模型同时部署，可用通过在模型库目录（如 `home/snowkylin/saved_model_files`）中配置 [models.config](./models.config) 文件实现配置。


## Keras Sequential 模式模型的部署 

由于 Sequential 模式的输入和输出都很固定，因此这种类型的模型很容易部署，无需其他额外操作。例如，要将使用 SavedModel 导出的 MNIST 手写体识别模型 （使用 Keras Sequential 模式建立）以 MLP 的模型名在 8501 端口进行部署，可以直接使用以下命令：

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved"  # 文件夹绝对地址根据自身情况填写，无需加入版本号


## 自定义 Keras 模型的部署 

使用继承 `tf.keras.Model` 类建立的自定义 Keras 模型的自由度相对更高。因此当使用 TensorFlow Serving 部署模型时，对导出的 SavedModel 文件也有更多的要求：

+ 需要导出到 SavedModel 格式的方法（比如 `call` ）不仅需要使用 `@tf.function` 修饰，还要在修饰时指定 `input_signature` 参数，以显式说明输入的形状。该参数传入一个由 `tf.TensorSpec` 组成的列表，指定每个输入张量的形状和类型。例如，对于 MNIST 手写体数字识别，我们的输入是一个 `[None, 28, 28, 1]` 的四维张量（ None 表示第一维即 Batch Size 的大小不固定），此时我们可以将模型的 `call` 方法做以下修饰：

```python
class MLP(tf.keras.Model):
    ...

    @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
    def call(self, inputs):
        ...
```

+ 在将模型使用 `tf.saved_model.save` 导出时，需要通过 `signature` 参数提供待导出的函数的签名（Signature）。简单说来，由于自定义的模型类里可能有多个方法都需要导出，因此，需要告诉 TensorFlow Serving 每个方法在被客户端调用时分别叫做什么名字。例如，如果我们希望客户端在调用模型时使用 `call` 这一签名来调用` model.call` 方法时，我们可以在导出时传入 `signature` 参数，以 `dict` 的键值对形式告知导出的方法对应的签名：

```python
model = MLP()
...
tf.saved_model.save(model, "saved_with_signature/1", signatures={"call": model.call})
```

以上两步均完成后，即可使用以下命令部署：

    tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=MLP \
        --model_base_path="/home/.../.../saved_with_signature"  # 修改为自己模型的绝对地址

## 在客户端调用以 TensorFlow Serving 部署的模型

TensorFlow Serving 支持以 gRPC 和 RESTful API 调用以 TensorFlow Serving 部署的模型。这里主要介绍较为通用的 RESTful API 方法。

RESTful API 以标准的 HTTP POST 方法进行交互，请求和回复均为 JSON 对象。为了调用服务器端的模型，我们在客户端向服务器发送以下格式的请求：

服务器 URI： `http://服务器地址:端口号/v1/models/模型名:predict`

请求内容：

    {
        "signature_name": "需要调用的函数签名（Sequential模式不需要）",
        "instances": 输入数据
    }

回复为：

    {
        "predictions": 返回值
    }




