# 自定义层、损失函数和评估指标

我们不仅可以继承 `tf.keras.Model` 编写自己的模型类，也可以继承 t`f.keras.layers.Layer` 编写自己的层。

## 自定义层

自定义层需要继承 `tf.keras.layers.Layer` 类，并重写 `__init__` 、 `build` 和 `call` 三个方法，如下所示：

```python
class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # 初始化代码

    def build(self, input_shape):     # input_shape 是一个 TensorShape 类型对象，提供输入的形状
        # 在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状
        # 而不需要使用者额外指定变量形状。
        # 如果已经可以完全确定变量的形状，也可以在__init__部分创建变量
        self.variable_0 = self.add_weight(...)
        self.variable_1 = self.add_weight(...)

    def call(self, inputs):
        # 模型调用的代码（处理输入并返回输出）
        return output
```

例如，如果我们要自己实现一个全连接层（ `tf.keras.layers.Dense` ），可以按如下方式编写。此代码在 `build` 方法中创建两个变量，并在 `call` 方法中使用创建的变量进行运算：

```python
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_weight(name='w',
            shape=[input_shape[-1], self.units], initializer=tf.zeros_initializer())
        self.b = self.add_weight(name='b',
            shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred
```

在定义模型的时候，我们便可以如同 Keras 中的其他层一样，调用我们自定义的层 `LinearLayer`：

```python
class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = LinearLayer(units=1)

    def call(self, inputs):
        output = self.layer(inputs)
        return output
```

## 自定义损失函数和评估指标

自定义损失函数需要继承 `tf.keras.losses.Loss` 类，重写 `call` 方法即可，输入真实值 `y_true` 和模型预测值 `y_pred` ，输出模型预测值和真实值之间通过自定义的损失函数计算出的损失值。下面的示例为均方差损失函数：

```python
class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
```

自定义评估指标需要继承 `tf.keras.metrics.Metric` 类，并重写 `__init__` 、 `update_state` 和 `result` 三个方法。下面的示例对 `SparseCategoricalAccuracy` 评估指标类做了一个简单的重实现：

```python
class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total
```