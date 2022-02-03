import tensorflow as tf


def usage1():
    '''result:
    tf.Tensor(0.015389681, shape=(), dtype=float32)
    tf.Tensor([0. 0.], shape=(2,), dtype=float32)
    (2, 2)
    <dtype: 'float32'>
    [[1 2]
     [3 4]]
    '''
    print("========== data structure: ")

    # 定义一个随机数（标量）
    random_float = tf.random.uniform(shape=())
    print(random_float)

    # 定义一个零向量（含有2个元素）
    zero_vector = tf.zeros(shape=(2))
    print(zero_vector)

    # 定义两个2*2的常量矩阵
    A = tf.constant([[1., 2.], [3., 4.]])
    B = tf.constant([[5., 6.], [7., 8.]])
    print(A.shape)  # 查看矩阵的形状（各维度大小，如长、宽、高）
    print(A.dtype)  # 查看矩阵的类型
    print(A.numpy())  # 转为numpy数值，查看矩阵的值

    return A, B


def usage2(A: tf.Tensor, B: tf.Tensor):
    '''result:
    tf.Tensor(
    [[ 6.  8.]
     [10. 12.]], shape=(2, 2), dtype=float32) 
    tf.Tensor(
    [[19. 22.]
     [43. 50.]], shape=(2, 2), dtype=float32)
    '''
    print("========== data operation: ")

    C = tf.add(A, B)  # 计算矩阵和，要求矩阵元素类型相同（int32 无法和 float32 相加）
    D = tf.matmul(A, B)  # 计算矩阵乘积
    print(C, D)


def gradient1():
    '''计算 y = x**2 在 x=3.0 处的导数。

    Result:
    y = tf.Tensor(9.0, shape=(), dtype=float32) 
    y_grad = tf.Tensor(6.0, shape=(), dtype=float32)
    '''
    print("========== auto gradient: ")

    # 定义并初始化一个变量
    x = tf.Variable(initial_value=3.)

    with tf.GradientTape() as tape:
        y = tf.square(x)
    y_grad = tape.gradient(y, x)   # 计算x=3.0处，y关于x的导数；此时 tape依然可用
    print(y, y_grad)


def gradient2():
    '''计算损失函数 L=（X w + b - y）**2 在 w = (1., 2.)^T 及 b = 1.0时对 w, b的偏导数。

    Result:
    L = tf.Tensor(125.0, shape=(), dtype=float32) 
    w_grad = tf.Tensor([[ 70.] [100.]], shape=(2, 1), dtype=float32) 
    b_grad = tf.Tensor(30.0, shape=(), dtype=float32)
    '''
    print("========== partial gradient: ")

    # 定义常量矩阵
    x = tf.constant([[1., 2.], [3, 4]])
    y = tf.constant([[1.], [2.]])

    # 定义参数
    w = tf.Variable(initial_value=[[1.], [2.]])
    b = tf.Variable(initial_value=1.)

    # 定义函数
    with tf.GradientTape() as tape:
        L = tf.reduce_sum(tf.square(tf.matmul(x, w) + b - y))

    # 求偏导数
    w_grad, b_grad = tape.gradient(L, [w, b])
    print(L, w_grad, b_grad)


if __name__ == "__main__":
    # A, B = usage1()
    # usage2(A, B)

    # gradient1()
    gradient2()
