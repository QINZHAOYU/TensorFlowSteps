# TensorFlow 常用模块

+ [tf.train.Checkpoint：变量的保存与恢复](#`tf.train.Checkpoint`变量的保存与恢复)
+ [TensorBoard：训练过程可视化](#TensorBoard训练过程可视化)
    + [实时查看参数变化情况](#实时查看参数变化情况)
    + [查看Graph和Profile信息](#查看Graph和Profile信息)





## `tf.train.Checkpoint` ：变量的保存与恢复 

Checkpoint 只保存模型的参数，不保存模型的计算过程，因此一般用于在具有模型源代码的时候恢复之前训练好的模型参数。

很多时候，我们希望在模型训练完成后能将训练好的参数（变量）保存起来。在需要使用模型的其他地方载入模型和参数，就能直接得到训练好的模型。

TensorFlow 提供了 `tf.train.Checkpoint` 这一强大的变量保存与恢复类，可以使用其 `save()` 和 `restore()` 方法将 TensorFlow 中所有包含 Checkpointable State 的对象进行保存和恢复。  
具体而言，`tf.keras.optimizer` 、 `tf.Variable` 、 `tf.keras.Layer` 或者 `tf.keras.Model` 实例都可以被保存。其使用方法非常简单，我们首先声明一个 Checkpoint：

```python
checkpoint = tf.train.Checkpoint(model=model)
```

`tf.train.Checkpoint()` 接受的初始化参数比较特殊，是一个 `**kwargs `。具体而言，是一系列的键值对，键名可以随意取(在恢复变量的时候，我们还将使用这一键名)，值为需要保存的对象。例如，如果我们希望保存一个继承 `tf.keras.Model` 的模型实例 `model` 和一个继承 `tf.train.Optimizer` 的优化器 `optimizer` ，我们可以这样写：

```python
checkpoint = tf.train.Checkpoint(myAwesomeModel=model, myAwesomeOptimizer=optimizer)
```

接下来，当模型训练完成需要保存的时候，使用：

```python
checkpoint.save(save_path_with_prefix)
```

`save_path_with_prefix` 是保存文件的目录 + 前缀。例如，在源代码目录建立一个名为 save 的文件夹并调用一次 `checkpoint.save('./save/model.ckpt')` ，我们就可以在可以在 save 目录下发现名为 `checkpoint` 、 `model.ckpt-1.index` 、 `model.ckpt-1.data-00000-of-00001` 的三个文件，这些文件就记录了变量信息。`checkpoint.save()` 方法可以运行多次，每运行一次都会得到一个.index 文件和.data 文件，序号依次累加。

当在其他地方需要为模型重新载入之前保存的参数时，需要再次实例化一个 checkpoint，同时保持键名的一致。再调用 checkpoint 的 restore 方法。就像下面这样：

```python
model_to_be_restored = MyModel()    # 待恢复参数的同一模型
checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)   # 键名保持为“myAwesomeModel”
checkpoint.restore(save_path_with_prefix_and_index)
```

即可恢复模型变量。 `save_path_with_prefix_and_index` 是之前保存的文件的目录 + 前缀 + 编号。例如，调用 `checkpoint.restore('./save/model.ckpt-1')` 就可以载入前缀为 model.ckpt ，序号为 1 的文件来恢复模型。

当保存了多个文件时，我们往往想载入最近的一个。可以使用 `tf.train.latest_checkpoint(save_path)` 这个辅助函数返回目录下最近一次 checkpoint 的文件名。例如如果 save 目录下有 model.ckpt-1.index 到 `model.ckpt-10.index` 的 10 个保存文件， `tf.train.latest_checkpoint('./save')` 即返回 `./save/model.ckpt-10` 。

总体而言，恢复与保存变量的典型代码框架如下：

```python
# train.py 模型训练阶段

model = MyModel()
# 实例化Checkpoint，指定保存对象为model（如果需要保存Optimizer的参数也可加入）
checkpoint = tf.train.Checkpoint(myModel=model)
# ...（模型训练代码）
# 模型训练完毕后将参数保存到文件（也可以在模型训练过程中每隔一段时间就保存一次）
checkpoint.save('./save/model.ckpt')

#------------------------------------------------------------------

# test.py 模型使用阶段

model = MyModel()
checkpoint = tf.train.Checkpoint(myModel=model)             # 实例化Checkpoint，指定恢复对象为model
checkpoint.restore(tf.train.latest_checkpoint('./save'))    # 从文件恢复模型参数
# ... 模型使用代码
```

[MLP模型参数保存示例](./CheckPointMlp.py)

在模型的训练过程中，我们往往每隔一定步数保存一个 Checkpoint 并进行编号。不过很多时候我们会有这样的需求：

+ 在长时间的训练后，程序会保存大量的 Checkpoint，但我们只想保留最后的几个 Checkpoint；
+ Checkpoint 默认从 1 开始编号，每次累加 1，但我们可能希望使用别的编号方式（例如使用当前 Batch 的编号作为文件编号）。

我们可以使用 TensorFlow 的 `tf.train.CheckpointManager` 来实现以上需求。具体而言，在定义 Checkpoint 后接着定义一个 CheckpointManager：

```python
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory='./save', checkpoint_name='model.ckpt', max_to_keep=k)
```

此处， `directory` 参数为文件保存的路径， `checkpoint_name` 为文件名前缀（不提供则默认为 `ckpt` ）， `max_to_keep` 为保留的 Checkpoint 数目。


## TensorBoard：训练过程可视化 

TensorBoard 就是一个能够帮助我们将训练过程可视化的工具。

### 实时查看参数变化情况 

首先在代码目录下建立一个文件夹（如 `./tensorboard` ）存放 TensorBoard 的记录文件，并在代码中实例化一个记录器：

```python
summary_writer = tf.summary.create_file_writer('./tensorboard')     # 参数为记录文件所保存的目录
```

接下来，当需要记录训练过程中的参数时，通过 `with` 语句指定希望使用的记录器，并对需要记录的参数（一般是 scalar）运行 `tf.summary.scalar(name, tensor, step=batch_index)` ，即可将训练过程中参数在 `step` 时候的值记录下来。这里的 `step` 参数可根据自己的需要自行制定，一般可设置为当前训练过程中的 `batch` 序号。整体框架如下：

```python
summary_writer = tf.summary.create_file_writer('./tensorboard')
# 开始模型训练
for batch_index in range(num_batches):
    # ...（训练代码，当前batch的损失值放入变量loss中）
    with summary_writer.as_default():                               # 希望使用的记录器
        tf.summary.scalar("loss", loss, step=batch_index)
        tf.summary.scalar("MyScalar", my_scalar, step=batch_index)  # 还可以添加其他自定义的变量
```

每运行一次 `tf.summary.scalar()` ，记录器就会向记录文件中写入一条记录。除了最简单的标量（scalar）以外，TensorBoard 还可以对其他类型的数据（如图像，音频等）进行可视化，详见 TensorBoard 文档 。

当我们要对训练过程可视化时，在代码目录打开终端（如需要的话进入 TensorFlow 的 conda 环境），运行:

```python
tensorboard --logdir=./tensorboard
```

然后使用浏览器访问命令行程序所输出的网址（一般是 http://name-of-your-computer:6006），即可访问 TensorBoard 的可视界面。

<img src="./imgs/tensorboard.png">

默认情况下，TensorBoard 每 30 秒更新一次数据。不过也可以点击右上角的刷新按钮手动刷新。

TensorBoard 的使用有以下注意事项：

+ 如果需要重新训练，需要删除掉记录文件夹内的信息并重启 TensorBoard（或者建立一个新的记录文件夹并开启 TensorBoard， `--logdir` 参数设置为新建立的文件夹）；
+ 记录文件夹目录保持全英文。

### 查看 Graph 和 Profile 信息

我们可以在训练时使用 `tf.summary.trace_on` 开启 `Trace`，此时 TensorFlow 会将训练时的大量信息（如计算图的结构，每个操作所耗费的时间等）记录下来。在训练完成后，使用 `tf.summary.trace_export` 将记录结果输出到文件。

```python
tf.summary.trace_on(graph=True, profiler=True)  # 开启Trace，可以记录图结构和profile信息
# 进行训练
with summary_writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)    # 保存Trace信息到文件
```

之后，我们就可以在 TensorBoard 中选择 “Profile”，以时间轴的方式查看各操作的耗时情况。如果使用了 `tf.function` 建立了计算图，也可以点击 “Graphs” 查看图结构。










