import numpy as np
import tensorflow as tf
#申明常量
x = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
y = tf.constant([[10.0],[20.0]])
#模型类定义
class Linear(tf.keras.Model):
    #初始化模型所需要的层,dense是全联接层
    def __init__(self):
        super().__init__()#super是调用父类函数
        self.dense=tf.keras.layers.Dense(
            units=1,#神经元的个数
            activation=None,#激活函数,不用激活函数大都是线性关系
            kernel_initializer=tf.zeros_initializer(),#权重系数（w1,w2,w3)
            bias_initializer=tf.zeros_initializer()#偏置系数b
        )
    def call(self,input):
        output = self.dense(input)
        return output
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:#开录音器开始记录
        y_pred = model(x)
        #对所有样本的误差做均值
        loss = tf.reduce_mean(tf.square(y_pred-y))#调用模型y_pred = model(x)而不是显示写出y_pred = a*x + b
    grads = tape.gradient(loss,model.variables)#梯度  [[w1,w2,w3],b]
    optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
print(model.variables)

#训练的结果是 y = x1*0.40784496+x2*1.191065+x3*1.9742855+0.78322077