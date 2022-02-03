from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

import platform 
plat = platform.system()

import matplotlib
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


X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class LinearModel(tf.keras.Model):
    '''通过 tensorflow 2 的 keras 接口风格实现线性回归模型。
    '''

    def __init__(self):
        '''重载模型初始化方案。
        '''
        super().__init__()  # 父类初始化

        # 配置全连接层
        self.dense = tf.keras.layers.Dense(
            units=1,  # 输出张量的维度（每个参数有几份）
            activation=None,  # 激活函数
            kernel_initializer=tf.zeros_initializer(),  # 参数初始化方式
            bias_initializer=tf.zeros_initializer()  # 偏置初始化方式
        )

    def call(self, input):
        '''重载模型调用接口。

        `tf.keras.Model` 这一父类已经包含 `__call__() `的定义。 
        `__call__()` 中主要调用了 `call()` 方法。通过继承 `tf.keras.Model` 并重载 `call()` 方法，
        即可在保持 keras 结构的同时加入模型调用的代码。
        '''
        output = self.dense(input)
        return output


def ModelWrapper():
    # 声明一个线性回归模型
    model = LinearModel()

    # 定义一个梯度下降优化器
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    for i in range(100):
        with tf.GradientTape() as tape:
            # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
            y_pred = model(X)
            loss = tf.reduce_sum(tf.square(y_pred - y))

        # 使用 model.variables 这一属性直接获得模型中的所有变量
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    return model.variables


def predict(X: tf.constant, vars: np.array, const: np.array):
    x_np = X.numpy()
    y_pred = x_np.dot(vars) + const
    return y_pred


if __name__ == "__main__":
    vars, const = ModelWrapper()  # 线性回归模型参数包括系数（w_i）和常量(b)
    y_pred = predict(X, vars.numpy(), const.numpy())
    print(y.numpy(), y_pred)
    plot(y_raw=y.numpy(), y_tf=y_pred)
