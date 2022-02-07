# TensorFlow 模型导出

为了将训练好的机器学习模型部署到各个目标平台（如服务器、移动端、嵌入式设备和浏览器等），我们的第一步往往是将训练好的整个模型完整导出（序列化）为一系列标准格式的文件。在此基础上，我们才可以在不同的平台上使用相对应的部署工具来部署模型文件。  
TensorFlow 提供了统一模型导出格式 SavedModel，使得我们训练好的模型可以以这一格式为中介，在多种不同平台上部署，这是我们在 TensorFlow 2 中主要使用的导出格式。  
同时，基于历史原因，Keras 的 Sequential 和 Functional 模式也有自有的模型导出格式。

## 使用 SavedModel 完整导出模型

Checkpoint 可以帮助我们保存和恢复模型中参数的权值。而作为模型导出格式的 SavedModel 则更进一步，其包含了一个 TensorFlow 程序的完整信息：不仅包含参数的权值，还包含计算的流程（即计算图）。当模型导出为 SavedModel 文件时，无须模型的源代码即可再次运行模型，这使得 SavedModel 尤其适用于模型的分享和部署。  
TensorFlow Serving（服务器端部署模型）、TensorFlow Lite（移动端部署模型）以及 TensorFlow.js 都会用到这一格式。

Keras 模型均可方便地导出为 SavedModel 格式。不过需要注意的是，因为 SavedModel 基于计算图，所以对于使用继承 `tf.keras.Model` 类建立的 Keras 模型，其需要导出到 SavedModel 格式的方法（比如 `call` ）都需要使用 `@tf.function` 修饰。  
假设我们有一个名为 `model` 的 Keras 模型，使用下面的代码即可将模型导出为 SavedModel：

```python
tf.saved_model.save(model, "保存的目标文件夹名称")
```

在需要载入 SavedModel 文件时，使用：

```python
model = tf.saved_model.load("保存的目标文件夹名称")
```

对于使用继承 `tf.keras.Model` 类建立的 Keras 模型 `model` ，使用 SavedModel 载入后将无法使用 `model()` 直接进行推断，而需要使用 `model.call()` 进行显式调用。

[模型导出导入用例](./ModelSavingAndLoadingUsages.py)

