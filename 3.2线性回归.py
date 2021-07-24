import numpy as np
import tensorflow as tf
x_raw = np.array([2013,2014,2015,2016,2017],dtype=np.float32)
y_raw = np.array([12000,14000,15000,16500,17500],dtype=np.float32)

x = (x_raw-x_raw.min())/(x_raw.max()-x_raw.min())#归一化
y = (y_raw-y_raw.min())/(y_raw.max()-y_raw.min())

#tensorflow 线性回归
#声明两个常量x,y
x = tf.constant(x)
y = tf.constant(y)
#声明两个变量a,b
a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

#开始训练
num_epoch = 10000
#声明了一个梯度下降 优化器 （Optimizer），其学习率为 5e-4。
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    #打开录像器来记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        #所预测的y
        y_pred = a * x + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
        #Tensorflow自动计算算是函数关于自变量的梯度
        # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
        grads = tape.gradient(loss, variables)
        #grade是梯度[a_grade,b_grade]
        #[a,b]
        #[a_grade,a],[b_grade,b]
        # TensorFlow自动根据梯度更新参数
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
print(a,b)