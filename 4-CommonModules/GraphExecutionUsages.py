import tensorflow as tf
import numpy as np
import time


gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


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


num_batches = 1000
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()
model = CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


@tf.function
def train_one_step(X, y):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        # 注意这里使用了TensorFlow内置的tf.print()。@tf.function不支持Python内置的print方法
        # tf.print("loss", loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


def training_graph():
    start_time = time.time()
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        train_one_step(X, y)
    end_time = time.time()
    print(end_time - start_time)


def training_eager():
    '''模型训练。
    '''
    start = time.time()
    for batch_index in range(num_batches):  # num_batches，电脑发热。。。
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
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    training_graph()  # time cost: 12.220665693283081 (s)
    # training_eager()   # time cost: 16.2248592376709 (s)
