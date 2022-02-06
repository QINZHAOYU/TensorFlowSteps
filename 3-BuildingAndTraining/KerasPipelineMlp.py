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


def build_model_by_sequential():
    '''通过 keras pipeline 中 tf.keras.models.Sequential() 实现mlp。

    通过一个层的列表, 就能快速地建立一个层叠结构的 tf.keras.Model 模型。
    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax()
    ])
    return model


def build_model_by_functional():
    '''通过 keras pipeline 中 functional api 实现mlp。

    例如多输入 / 输出或存在参数共享的非层叠结构模型，
    将层作为可调用的对象并返回张量, 并将输入向量和输出向量提供给 tf.keras.Model 的 inputs 和 outputs 参数。
    '''
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(units=10)(x)
    outputs = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def training(model, learning_rate, num_epochs, batch_size, data_loader):
    # 激活模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    # 拟合数据，训练模型
    model.fit(data_loader.train_data, data_loader.train_label,
              epochs=num_epochs, batch_size=batch_size)


def evaluating(model, data_loader):
    print(model.evaluate(data_loader.test_data, data_loader.test_label))


if __name__ == "__main__":
    # model = build_model_by_sequential()
    model = build_model_by_functional()

    learning_rate = 0.001
    num_epochs = 5
    batch_size = 10
    data_loader = MNISTLoader()

    training(model, learning_rate, num_epochs, batch_size, data_loader)
    evaluating(model, data_loader)
