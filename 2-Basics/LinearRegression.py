import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('TkAgg')


def data_preprocess():
    '''生成数据集并归一化处理。
    '''
    X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
    y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

    X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
    return X, y


def predict(X, a, b):
    '''根据线性回归模型，预报结果。 
    '''
    y_pred = a * X + b
    return y_pred


def plot(**series):
    '''展示拟合结果。
    '''
    for key, data in series.items():
        x = range(len(data))
        plt.plot(x, data, ls="-", lw=2, label=str(key))
    plt.legend()
    plt.show()


def LR_NumPy(X, y):
    '''采用 NumPy 实现房价的线性回归建模 y = a x + b。

    Result: 
    a = 0.9763702027797713 
    b = 0.05756498835608132
    '''
    print("========== lr model by numpy: ")

    a, b = 0, 0
    num_epoch = 10000
    learning_rate = 5e-4

    for e in range(num_epoch):
        # 手动计算损失函数关于自变量（模型参数）的梯度
        y_pred = a * X + b
        grad_a = 2 * (y_pred - y).dot(X)  # 一维数组，内积为标量
        grad_b = 2 * (y_pred - y).sum()   # 一维数组，求和

        # 更新参数
        a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

    print(a, b)
    return a, b


def LR_TF2(X, y):
    '''采用 tensorflow 2 实现房价的线性回归建模 y = a x + b.

    Result:
    a = 0.97637 
    b = 0.057565063
    '''
    print("========== lr model by tensorflow2: ")
    X_t = tf.constant(X)
    y_t = tf.constant(y)

    # 定义模型参数
    a = tf.Variable(initial_value=0.)
    b = tf.Variable(initial_value=0.)
    variables = [a, b]

    # 声明一个梯度下降优化器，指定学习率
    optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)

    num_epoch = 10000
    for e in range(num_epoch):
        # 使用tf.GradientTape()记录损失函数的梯度信息
        with tf.GradientTape() as tape:
            y_pred = a * X_t + b
            loss = tf.reduce_sum(tf.square(y_pred - y_t))
        # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
        grads = tape.gradient(loss, variables)
        # TensorFlow自动根据梯度更新参数
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    print(a.numpy(), b.numpy())
    return a.numpy(), b.numpy()


if __name__ == "__main__":
    X, y = data_preprocess()
    a, b = LR_NumPy(X, y)
    a_tf, b_tf = LR_TF2(X, y)
    y_pred_np = predict(X, a, b)
    y_pred_tf = predict(X, a_tf, b_tf)
    plot(y_raw=y, y_np=y_pred_np, y_tf=y_pred_tf)
