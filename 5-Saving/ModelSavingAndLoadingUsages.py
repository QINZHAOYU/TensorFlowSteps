from ast import arg
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="tensorflow model saving and loading usages")
parser.add_argument(
    'mode', choices=["custom", "sequential"], help="model building type")
parser.add_argument(
    'usage', choices=['load', 'save'], help="save or load model")
parser.add_argument('--num_epochs', default=1)
parser.add_argument('--batch_size', default=50)
parser.add_argument('--learning_rate', default=0.001)
args = parser.parse_args()


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


data_loader = MNISTLoader()


def build_mlp_by_sequential():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax()
    ])
    return model


class MLP(tf.keras.Model):
    def __init__(self):
        '''搭建并初始化模型。
        '''
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    @tf.function          # 将自定义模型的 call 修饰后才可以保存模型。
    def call(self, inputs):       # [batch_size, 28, 28, 1]
        '''调用模型，预报结果。 
        '''
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)        # [batch_size, 100]
        x = self.dense2(x)        # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


def compile_and_fit(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(data_loader.train_data, data_loader.train_label,
              epochs=args.num_epochs, batch_size=args.batch_size)


def save(model, path="./5-Saving/save"):
    tf.saved_model.save(model, path)


def load(path="./5-Saving/save"):
    return tf.saved_model.load(path)


def test(model):
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // args.batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * \
            args.batch_size, (batch_index + 1) * args.batch_size

        if args.mode == "sequential":
            # sequential 模型无法显式调用 call()。
            y_pred = model(data_loader.test_data[start_index: end_index])
        if args.mode == "custom":
            # 对于自定义类，load 载入后需要显式调用
            y_pred = model.call(data_loader.test_data[start_index: end_index])

        sparse_categorical_accuracy.update_state(
            y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())


if __name__ == "__main__":
    if args.mode == "sequential":
        model = build_mlp_by_sequential()
    if args.mode == "custom":
        model = MLP()

    compile_and_fit(model)

    if args.usage == "save":
        save(model)
        print("model saved.")
    if args.usage == "load":
        model = load()
        print("model loaded, test result:")
        test(model)
