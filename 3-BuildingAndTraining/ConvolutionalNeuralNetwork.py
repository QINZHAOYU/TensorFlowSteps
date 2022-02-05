from pyexpat import model
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def conv2d_demo():
    '''Conv2D卷积层示例。

    Result:
    [ 6.  5. -2.  1.  2.]
    [ 3.  0.  3.  2. -2.]
    [ 4.  2. -1.  0.  0.]
    [ 2.  1.  2. -1. -3.]
    [ 1.  1.  1.  3.  1.]
    '''
    image = np.array([[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 2, 1, 0],
        [0, 0, 2, 2, 0, 1, 0],
        [0, 1, 1, 0, 2, 1, 0],
        [0, 0, 2, 1, 1, 0, 0],
        [0, 2, 1, 1, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]], dtype=np.float32)

    # np.expand_dims(arr, axis), 在指定轴axis上增加数组arr的一个维度.
    # 若 a.shape 是 （m, n, c）, 则：
    # b = np.expand_dims(a, axis=0), b.shape (1, m, n, c);
    # c = np.expand_dims(a, axis=1), c.shape (m, 1, n, c);
    # d = np.expand_dims(a, axis=2), d.shape (m, n, 1, c);
    # e = np.expand_dims(a, axis=3), e.shape (m, n, c, 1).
    image = np.expand_dims(image, axis=-1)
    W = np.array([[
        [0, 0, -1],
        [0, 1, 0],
        [-2, 0, 2]
    ]], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=[3, 3],
            kernel_initializer=tf.constant_initializer(W),
            bias_initializer=tf.constant_initializer(b)
        )
    ])

    output = model(image)
    print(tf.squeeze(output))


class CNN(tf.keras.Model):
    '''使用 tf.keras 自定义实现卷积神经网络。
    '''

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu,  # 激活函数
            strides=1,              # 滑动步长
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        # self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


'''
from MultilayerPerceptron import MNISTLoader

这里会导致gpu分配失败: Physical devices cannot be modified after being initialized。
原因是在模块 MultilayerPerceptron 中已经导入 tf2 并实例化了模型。
'''


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


def training(batch_size, data_loader, model):
    '''模型训练。
    '''
    num_epochs = 5
    learning_rate = 0.001

    # 定义一个Adam优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # `//`, floor division, a//b == floor(a/b)
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)

    for batch_index in range(3000):  # num_batches，电脑发热。。。
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            # 执行预报(?)
            y_pred = model(X)
            # 交叉熵函数计算损失（误差）
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        # 计算梯度
        grads = tape.gradient(loss, model.variables)
        # 应用梯度
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print("training is done")
    return model.variables


def evaluating(batch_size, data_loader, model):
    '''模型评估。

    Result: 
    test accuracy: 0.985100 (training steps 3000, while mlp run 30000 steps)
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


def MobielNetV2_demo():
    '''使用 Keras 中预定义的经典卷积神经网络结构.

    Failed.
    '''
    num_epoch = 5
    batch_size = 50
    learning_rate = 0.001

    dataset = tfds.load(
        "tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
    dataset = dataset.map(lambda img, label: (tf.image.resize(
        img, (224, 224)) / 255.0, label)).shuffle(1024).batch(batch_size)

    model = tf.keras.applications.MobileNetV2(weights=None, classes=5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for e in range(num_epoch):
        for images, labels in dataset:
            with tf.GradientTape() as tape:
                labels_pred = model(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y_true=labels, y_pred=labels_pred)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(
                grads, model.trainable_variables))
        print(labels_pred)


if __name__ == "__main__":
    # conv2d_demo()

    data_loader = MNISTLoader()
    batch_size = 10
    model = CNN()
    training(batch_size, data_loader, model)
    evaluating(batch_size, data_loader, model)

    # MobielNetV2_demo()
