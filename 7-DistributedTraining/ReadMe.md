#  TensorFlow 分布式训练

当我们拥有大量计算资源时，通过使用合适的分布式策略，我们可以充分利用这些计算资源，从而大幅压缩模型训练的时间。针对不同的使用场景，TensorFlow 在 `tf.distribute.Strategy` 中为我们提供了若干种分布式策略，使得我们能够更高效地训练模型。


## 单机多卡训练： `MirroredStrategy`

`tf.distribute.MirroredStrategy` 是一种简单且高性能的，数据并行的同步式分布式策略，主要支持多个 GPU 在同一台主机上训练。使用这种策略时，我们只需实例化一个 `MirroredStrategy` 策略:

```python
strategy = tf.distribute.MirroredStrategy()
```

可以在参数中指定设备，如:

```python
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
```
即指定只使用第 0、1 号 GPU 参与分布式策略。

并将模型构建的代码放入 `strategy.scope()` 的上下文环境中:

```python
with strategy.scope():
    # 模型构建代码
```
使用 `MirroredStrategy` 后，模型训练的速度有了大幅度的提高。在所有显卡性能接近的情况下，训练时长与显卡的数目接近于反比关系。

`MirroredStrategy` 的步骤如下：

+ 训练开始前，该策略在所有 N 个计算设备上均各复制一份完整的模型；
+ 每次训练传入一个批次的数据时，将数据分成 N 份，分别传入 N 个计算设备（即数据并行）；
+ N 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据的梯度；
+ 使用分布式计算的 All-reduce 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都有了所有设备的梯度之和；
+ 使用梯度求和的结果更新本地变量（镜像变量）；
+ 当所有设备均更新本地变量后，进行下一轮训练（即该并行策略是同步的）。

默认情况下，TensorFlow 中的 `MirroredStrategy` 策略使用 NVIDIA NCCL 进行 All-reduce 操作。


## 多机训练： `MultiWorkerMirroredStrategy`

多机训练的方法和单机多卡类似，将 `MirroredStrategy` 更换为适合多机训练的 `MultiWorkerMirroredStrategy` 即可。不过，由于涉及到多台计算机之间的通讯，还需要进行一些额外的设置。具体而言，需要设置环境变量 `TF_CONFIG` ，示例如下:

```python
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:20000", "localhost:20001"]
    },
    'task': {'type': 'worker', 'index': 0}
})


'''json module

# 将python对象格式化为json对象（字符串）
json_obj = json.dumps(py_dict_obj, ensure_ascii=False)

# 将python对象按格式写入json文件
f = open('output.json', 'w', encoding='utf-8')
json.dump(py_dict_obs, f, ensure_ascii=False)
f.close()

# 将json对象解码为python对象（字典）
py_obj = json.loads(json_ojb)

# 加载json文件返回为python对象。
with open("out.json", "r", encoding="utf-8") as f:
    py_obj = json.load(f)
'''
```

`TF_CONFIG` 由 `cluster` 和 `task` 两部分组成：

+ `cluster` 说明了整个多机集群的结构和每台机器的网络地址（IP + 端口号）。对于每一台机器，`cluster` 的值都是相同的；
+ task 说明了当前机器的角色。例如， `{'type': 'worker', 'index': 0}` 说明当前机器是 `cluster` 中的第 0 个 `worker`（即 `localhost:20000` ）。每一台机器的 `task` 值都需要针对当前主机进行分别的设置。

以上内容设置完成后，在所有的机器上逐个运行训练代码即可。先运行的代码在尚未与其他主机连接时会进入监听状态，待整个集群的连接建立完毕后，所有的机器即会同时开始训练。

在所有机器性能接近的情况下，训练时长与机器的数目接近于反比关系。