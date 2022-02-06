# TensorFlow 常用模块

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