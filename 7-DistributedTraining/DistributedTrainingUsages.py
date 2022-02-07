import tensorflow as tf
import tensorflow_datasets as tfds


# 查看当前主机上某种特定运算设备
gpus = tf.config.list_physical_devices(device_type='GPU')
cpus = tf.config.list_physical_devices(device_type='CPU')
print("============ static devices: ")
print(gpus, cpus)
# 建立两个显存均为 1GB 的虚拟 GPU
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
     tf.config.LogicalDeviceConfiguration(memory_limit=1024)])


num_epochs = 5
batch_size_per_replica = 64
learning_rate = 0.001

# 实例化镜像策略
strategy = tf.distribute.MirroredStrategy()
print('============ Number of devices: %d' %
      strategy.num_replicas_in_sync)  # 输出设备数量(2)
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync


def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label


def get_dataset():
    '''使用 TensorFlow Datasets 载入猫狗分类数据集。
    '''
    dataset = tfds.load(
        "cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
    dataset = dataset.map(resize).shuffle(1024).batch(batch_size)
    return dataset


def distributed_training(dataset):
    '''单机多卡训练 MobileNetV2 模型。
    '''
    with strategy.scope():
        model = tf.keras.applications.MobileNetV2(weights=None, classes=2)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy]
        )
    # model.fit(dataset, epochs=num_epochs)   #  Failed.
    return model


if __name__ == "__main__":
    dataset = get_dataset()
    model = distributed_training(dataset)
