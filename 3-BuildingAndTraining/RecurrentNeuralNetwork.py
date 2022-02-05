import tensorflow as tf
import numpy as np


class DataLoader():
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt',
                                       origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index+seq_length])
            next_char.append(self.text[index+seq_length])
        # [batch_size, seq_length], [num_batch]
        return np.array(seq), np.array(next_char)


class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cell = tf.keras.layers.LSTMCell(units=256)  # 使用 LSTMCell 单元
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False):
        # [batch_size, seq_length, num_chars]

        # tf.one_hot(indices, depth, ...)，返回一个标识张量，
        # labels = [0, 2, -1, 1]
        # res = tf.one_hot(indices=labels, depth=4, on_value=1.0, off_value=0.0, axis=-1)
        # with tf.Session() as sess:
        #     print(sess.run(res))
        # Result:
        # [[ 1.  0.  0.  0.]
        #  [ 0.  0.  1.  0.]
        #  [ 0.  0.  0.  0.]
        #  [ 0.  1.  0.  0.]]
        inputs = tf.one_hot(inputs, depth=self.num_chars)
        state = self.cell.get_initial_state(
            batch_size=self.batch_size, dtype=tf.float32)   # 获得 RNN 的初始状态
        for t in range(self.seq_length):
            # 通过当前输入和前一时刻的状态，得到输出和当前时刻的状态
            output, state = self.cell(inputs[:, t, :], state)
        logits = self.dense(output)
        if from_logits:   # from_logits 参数控制输出是否通过 softmax 函数进行归一化
            return logits
        else:
            return tf.nn.softmax(logits)

    def predict(self, inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        # 调用训练好的RNN模型，预测下一个字符的概率分布
        logits = self(inputs, from_logits=True)
        # 使用带 temperature 参数的 softmax 函数获得归一化的概率分布值
        prob = tf.nn.softmax(logits / temperature).numpy()
        # 使用 np.random.choice 函数，在预测的概率分布 prob 上进行随机取样
        return np.array([np.random.choice(self.num_chars, p=prob[i, :]) for i in range(batch_size.numpy())])


def training(num_batches, seq_length, batch_size, learning_rate):
    data_loader = DataLoader()
    model = RNN(num_chars=len(data_loader.chars),
                batch_size=batch_size, seq_length=seq_length)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(seq_length, batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            # print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    return data_loader, model


def generate(data_loader, seq_length, model):
    X_, _ = data_loader.get_batch(seq_length, 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:      # 丰富度（即temperature）分别设置为从小到大的 4 个值
        X = X_
        print("diversity %f:" % diversity)
        for t in range(400):
            y_pred = model.predict(X, diversity)    # 预测下一个字符的编号
            print(data_loader.indices_char[y_pred[0]],
                  end='', flush=True)  # 输出预测的字符
            # 将预测的字符接在输入 X 的末尾，并截断 X 的第一个字符，以保证 X 的长度不变
            X = np.concatenate(
                [X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
        print("\n")


if __name__ == "__main__":
    num_batches = 1000
    seq_length = 40
    batch_size = 50
    learning_rate = 1e-3

    data_loader, model = training(
        num_batches, seq_length, batch_size, learning_rate)
    generate(data_loader, seq_length, model)
