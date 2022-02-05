# Keras Pipeline

在很多时候，我们只需要建立一个结构相对简单和典型的神经网络（比如 MLP 和 CNN），并使用常规的手段进行训练。这时，Keras 提供了另一套更为简单高效的内置方法来建立、训练和评估模型。


## Keras Sequential/Functional API 模式建立模型 

最典型和常用的神经网络结构是将一堆层按特定顺序叠加起来，那么，我们是不是只需要提供一个层的列表，就能由 Keras 将它们自动首尾相连，形成模型呢？Keras 的 Sequential API 正是如此。  
通过向 `tf.keras.models.Sequential()` 提供一个层的列表，就能快速地建立一个 `tf.keras.Model` 模型并返回：

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])
```

不过，这种层叠结构并不能表示任意的神经网络结构。为此，Keras 提供了 Functional API，帮助我们建立更为复杂的模型，例如多输入 / 输出或存在参数共享的模型。其使用方法是将层作为可调用的对象并返回张量，并将输入向量和输出向量提供给 `tf.keras.Model` 的 `inputs` 和 `outputs` 参数：

```python
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

## 使用 Keras Model 的 `compile` 、 `fit` 和 `evaluate` 方法训练和评估模型

当模型建立完成后，通过 `tf.keras.Model` 的 `compile` 方法配置训练过程：

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)
```

`tf.keras.Model.compile` 接受 3 个重要的参数：

+ `oplimizer` ：优化器，可从 `tf.keras.optimizers` 中选择；
+ `loss` ：损失函数，可从 `tf.keras.losses` 中选择；
+ `metrics` ：评估指标，可从 `tf.keras.metrics` 中选择。

接下来，可以使用 `tf.keras.Model` 的 `fit` 方法训练模型：

```python
model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)
```

`tf.keras.Model.fit` 接受 5 个重要的参数：

+ `x` ：训练数据；
+ `y` ：目标数据（数据标签）；
+ `epochs` ：将训练数据迭代多少遍；
+ `batch_size` ：批次的大小；
+ `validation_data` ：验证数据，可用于在训练过程中监控模型的性能。

最后，使用 `tf.keras.Model.evaluate` 评估训练效果，提供测试数据及标签即可：

```python
print(model.evaluate(data_loader.test_data, data_loader.test_label))
```