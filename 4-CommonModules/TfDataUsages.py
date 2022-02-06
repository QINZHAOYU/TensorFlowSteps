import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TKAgg")


def show(dataset, num=5):
    count = 1
    for image, label in dataset:
        plt.title(label.numpy())
        plt.imshow(image.numpy()[:, :, 0])
        plt.show()

        count += 1
        if count > num:
            break


def demo1():
    X = tf.constant([2013, 2014, 2015, 2016, 2017])
    Y = tf.constant([12000, 14000, 15000, 16500, 17500])

    # 也可以使用NumPy数组，效果相同
    # X = np.array([2013, 2014, 2015, 2016, 2017])
    # Y = np.array([12000, 14000, 15000, 16500, 17500])

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    for x, y in dataset:
        print(x.numpy(), y.numpy())


def demo2():
    '''mnist_dataset Result: 

    TensorSliceDataset element_spec=(
        TensorSpec(shape=(28, 28, 1), dtype=tf.float32, name=None), 
        TensorSpec(shape=(), dtype=tf.uint8, name=None))
    '''
    (train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
    train_data = np.expand_dims(train_data.astype(
        np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]

    mnist_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data, train_label))
    print(mnist_dataset)
    return mnist_dataset


def rot90(image, label):
    '''图片旋转 90 度。
    '''
    image = tf.image.rot90(image)
    return image, label


def demo3(mnist_dataset):
    '''数据集对象的预处理。
    '''
    mnist_dataset = mnist_dataset.map(rot90)
    return mnist_dataset


def demo4(mnist_dataset):
    '''将数据集划分批次，每个批次的大小为 4.
    '''
    mnist_dataset = mnist_dataset.batch(4)
    count = 1

    # image: [4, 28, 28, 1], labels: [4]
    for images, labels in mnist_dataset:
        fig, axs = plt.subplots(1, 4)
        for i in range(4):
            axs[i].set_title(labels.numpy()[i])
            axs[i].imshow(images.numpy()[i, :, :, 0])
        plt.show()

        count += 1
        if count > 5:
            break


def demo5(mnist_dataset):
    '''数据打散后再设置批次，缓存大小设置为 10000.
    '''
    mnist_dataset = mnist_dataset.shuffle(buffer_size=10000).batch(4)
    count = 1

    # image: [4, 28, 28, 1], labels: [4]
    for images, labels in mnist_dataset:
        fig, axs = plt.subplots(1, 4)
        for i in range(4):
            axs[i].set_title(labels.numpy()[i])
            axs[i].imshow(images.numpy()[i, :, :, 0])
        plt.show()

        count += 1
        if count > 5:
            break


def demo6(mnist_dataset):
    '''数据集并行化加速。
    '''
    # 开启预加载数据
    mnist_dataset = mnist_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    # 利用多核心的优势对数据进行并行化变换
    mnist_dataset = mnist_dataset.map(
        map_func=rot90, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return mnist_dataset


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


class demo7():
    def build_model_by_functional(self):
        '''通过 keras pipeline 中 functional api 实现mlp。
        '''
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(units=10)(x)
        outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def training(self, model, learning_rate, num_epochs, data_loader):
        # 激活模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy]
        )
        # 拟合数据，训练模型
        # model.fit(data_loader.train_data, data_loader.train_label,
        #           epochs=num_epochs, batch_size=batch_size)
        dataset = tf.data.Dataset.from_tensor_slices(
            (data_loader.train_data, data_loader.train_label))
        dataset = dataset.shuffle(buffer_size=10000).batch(
            4)  # 划分了数据集的批次(batch_size)
        model.fit(dataset, epochs=num_epochs)

    def evaluating(self, model, data_loader):
        print(model.evaluate(data_loader.test_data, data_loader.test_label))


if __name__ == "__main__":
    # demo1()

    # dataset = demo2()
    # # show(dataset)
    # dataset = demo3(dataset)
    # show(dataset)

    # demo4(dataset)
    # demo5(dataset)

    # dataset = demo6(dataset)
    # show(dataset)

    learning_rate = 0.001
    num_epochs = 5
    demo = demo7()
    data_loader = MNISTLoader()
    model = demo.build_model_by_functional()
    demo.training(model, learning_rate, num_epochs, data_loader)
    demo.evaluating(model, data_loader)
