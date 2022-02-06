import tensorflow as tf

# 获得当前主机上某种特定运算设备类型
gpus = tf.config.list_physical_devices(device_type='GPU')
cpus = tf.config.list_physical_devices(device_type='CPU')
print(gpus, cpus)

# 设置当前程序可见的设备范围
tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
# tf.config.set_visible_devices(devices=[], device_type='GPU')

# GPU 设置为仅在需要时申请显存空间
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(device=gpu, enable=True)

# 设置 TensorFlow 固定消耗 GPU:0 的 1GB 显存
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

# 建立两个显存均为 1GB 的虚拟 GPU
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
     tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
# LogicalDevice(name='/device:GPU:0', device_type='GPU'),
# LogicalDevice(name='/device:GPU:1', device_type='GPU')
virtual_multi_gpus = tf.config.list_logical_devices(device_type='GPU')
print(virtual_multi_gpus)
