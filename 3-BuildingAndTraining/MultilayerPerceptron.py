import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

import platform
plat = platform.system()
if plat == "Windows":
    matplotlib.use("TKAgg")


def plot(**series):
    '''展示拟合结果。
    '''
    for key, data in series.items():
        x = range(len(data))
        plt.plot(x, data, ls="-", lw=2, label=str(key))
    plt.legend()
    if plat == "Windows":
        plt.show()
    elif plat == "Linux":
        plt.savefig("result.png")


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


class MLP(tf.keras.Model):
    def __init__(self):
        '''搭建并初始化模型。
        '''
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):       # [batch_size, 28, 28, 1]
        '''调用模型，预报结果。 
        '''
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)        # [batch_size, 100]
        x = self.dense2(x)        # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


data_loader = MNISTLoader()
batch_size = 10
model = MLP()


def training():
    '''模型训练。

    Result: 
    tf.Variable 'mlp/dense/kernel:0' shape=(784, 100) dtype=float32, ...;
    tf.Variable 'mlp/dense_1/kernel:0' shape=(100, 10) dtype=float32, ...;
    tf.Variable 'mlp/dense_1/bias:0' shape=(10,) dtype=float32, ...;
    '''
    num_epochs = 5
    learning_rate = 0.001

    # 定义一个Adam优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # `//`, floor division, a//b == floor(a/b)
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)

    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            # 执行预报(?)
            y_pred = model(X)
            # 交叉熵函数计算损失（误差）
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        # 计算梯度
        grads = tape.gradient(loss, model.variables)
        # 应用梯度
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    return model.variables


def evaluating():
    '''模型评估。

    Result: 
    test accuracy: 0.974500
    '''
    # 实例化一个评估器
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):  # 分次输入预报和实测数据
        start_index, end_index = batch_index * \
            batch_size, (batch_index + 1) * batch_size
        # 模型预报
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        # 模型评估
        sparse_categorical_accuracy.update_state(
            y_true=data_loader.test_label[start_index: end_index],
            y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())


def predict(X: np.ndarray, model: MLP):
    '''模型预报。

    依然不知道如何调用模型识别图片中数字并返回？
    + model(X) ?
    + model.predict(X) ?
    '''
    pass


if __name__ == "__main__":
    training()
    evaluating()
