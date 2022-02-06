import tensorflow as tf
import numpy as np


class MNISTLoader():
    '''加载 mnist 数据集。
    '''

    def __init__(self):
        '''加载mnist 手写图片数据集。
        '''
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data,
                                              self.test_label) = mnist.load_data()

        # MNIST中的图像默认为uint8（0-255的数字）。
        # 将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道。
        self.train_data = np.expand_dims(self.train_data.astype(
            np.float32) / 255.0, axis=-1)        # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(
            np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]

        # 数据标签，数据类型转为整型。
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[
            0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        '''从数据集中随机取出batch_size个元素并返回。
        '''
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]


class MyDense(tf.keras.layers.Layer):
    '''通过继承 tf.keras.layers.Layer 类，并重写 __init__ 、 build 和 call 三个方法，
    实现自定义 全连接层 。
    '''

    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units  # 神经元数量
        self.activation = tf.keras.layers.Activation(activation)  # 激活函数

    def build(self, input_shape):
        # super().build(input_shape)
        # 定义权重值数量和初始化方式
        self.w = self.add_weight(name='w',
                                 shape=[input_shape[-1], self.units],
                                 initializer=tf.zeros_initializer()
                                 )
        # 定义偏置值和初始化方式
        self.b = self.add_weight(name='b',
                                 shape=[self.units],
                                 initializer=tf.zeros_initializer()
                                 )

    def call(self, inputs):
        y_pred = self.activation(tf.matmul(inputs, self.w) + self.b)
        return y_pred


def build_model_by_sequential():
    '''通过 keras pipeline 中 tf.keras.models.Sequential() 实现mlp。

    通过一个层的列表, 就能快速地建立一个层叠结构的 tf.keras.Model 模型。
    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        # MyDense(100, activation=tf.nn.relu), # 效果很差很差
        # tf.keras.layers.Dense(10),
        MyDense(10),  # 调用自定义层
        tf.keras.layers.Softmax()
    ])
    return model


class MyLoss(tf.keras.losses.Loss):
    '''通过继承 tf.keras.losses.Loss 类，重写 call 方法， 
    实现自定义 均方差 损失函数。
    '''

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        '''相比 sparse_categorical_crossentropy，效果很差很差。
        '''
        y_true = tf.cast(y_true, dtype=tf.float32)
        return tf.reduce_mean(tf.square(y_pred - y_true))


def loss(y_true, y_pred):
    '''自定义函数式 均方差 损失函数。

    相比 sparse_categorical_crossentropy，效果很差很差。
    '''
    y_true = tf.cast(y_true, dtype=tf.float32)
    return tf.reduce_mean(tf.square(y_pred - y_true))


class MyMetric(tf.keras.metrics.Metric):
    '''通过继承 tf.keras.metrics.Metric 类，并重写 __init__ 、 update_state 和 result 三个方法，
    实现自定义 稀疏分类准确性（稀疏分类准确性） 评估函数。

    效果很差。
    '''

    def __init__(self):
        super().__init__()
        self.total = self.add_weight(
            name="total", dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(
            name="count", dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(y_true, tf.argmax(
            y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count/self.total


def training(model, learning_rate, num_epochs, batch_size, data_loader):
    # 激活模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        # loss=loss,
        # loss=MyLoss(),
        # metrics=[tf.keras.metrics.sparse_categorical_accuracy],
        metrics=[MyMetric()]
    )
    # 拟合数据，训练模型
    model.fit(data_loader.train_data, data_loader.train_label,
              epochs=num_epochs, batch_size=batch_size)


def evaluating(model, data_loader):
    print(model.evaluate(data_loader.test_data, data_loader.test_label))


if __name__ == "__main__":
    model = build_model_by_sequential()

    learning_rate = 0.001
    num_epochs = 5
    batch_size = 10
    data_loader = MNISTLoader()

    training(model, learning_rate, num_epochs, batch_size, data_loader)
    evaluating(model, data_loader)
