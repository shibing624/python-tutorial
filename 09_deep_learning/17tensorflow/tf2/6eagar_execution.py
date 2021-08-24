# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import tensorflow as tf

print(tf.__version__)
# 在tensorflow2中默认使用Eager Execution
print(tf.executing_eagerly())

x = [[3.]]
m = tf.matmul(x, x)
print(m)

# tf.Tensor对象引用具体值
a = tf.constant([[1, 9], [3, 6]])
print(a)

# 支持broadcasting（广播：不同shape的数据进行数学运算）
b = tf.add(a, 2)
print(b)

# 支持运算符重载
print(a * b)

# 可以当做numpy数据使用
import numpy as np

s = np.multiply(a, b)
print(s)

# 转换为numpy类型
print(a.numpy())

c = tf.constant(0)
print('c:', c)
max_num = tf.convert_to_tensor(10)
print('max_num:', max_num)


def fizzbuzz(n):
    counter = tf.constant(0)
    n = tf.constant(n)
    for num in range(1, n.numpy() + 1):
        num = tf.constant(num)
        if int(num % 3) == 0 and int(num % 5) == 0:
            print('fizzbuzz')
        elif int(num % 3) == 0:
            print('fizz')
        elif int(num % 5) == 0:
            print('buzz')
        else:
            print(num.numpy())
        counter += 1


fizzbuzz(17)


w = tf.Variable([[1.0]])
# 用tf.GradientTape()记录梯度
with tf.GradientTape() as tape:
    loss = w*w*w
grad = tape.gradient(loss, w)  # 计算梯度，求导
print(grad)

# 导入mnist数据
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
# 数据转换
dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255, tf.float32),
     tf.cast(mnist_labels, tf.int64)))
# 数据打乱与分批次
dataset = dataset.shuffle(1000).batch(32)
# 使用Sequential构建一个卷积网络
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu',
                           input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])
# 展示数据
# 即使没有经过培训，也可以调用模型并在Eager Execution中检查输出
for images, labels in dataset.take(1):
    print("Logits: ", mnist_model(images[0:1]).numpy(),"labels: ", labels[0:1])

# 优化器与损失函数
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 按批次训练
# 虽然 keras 模型具有内置训练循环（fit 方法），但有时需要更多自定义设置。下面是一个用 eager 实现的训练循环示例：
loss_history = []
for (batch, (images, labels)) in enumerate(dataset.take(400)):
    if batch % 10 == 0:
        print('.', end='')
    with tf.GradientTape() as tape:
        # 获取预测结果
        logits = mnist_model(images, training=True)
        # 获取损失
        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    # 获取本批数据梯度
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    # 反向传播优化
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

# 绘图展示loss变化
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')

plt.savefig("a.png")