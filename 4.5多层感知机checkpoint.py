import numpy as np
import tensorflow as tf

#建立一个类来获取数据集
class MNISTloader():
    def __init__(self):
        #数据获取预处理
        mnist = tf.keras.datasets.mnist
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        #用load_data返回一个测试数据集和测试数据集
        (self.train_data,self.train_label),(self.test_data,self.test_label) = mnist.load_data()
        #np.expand_dims 用于扩展数组的形状
        self.train_data = np.expand_dims(self.train_data.astype(np.float32)/255.0,axis=-1)#归一化处理
        self.test_data = np.expand_dims(self.test_data.astype(np.float32)/255.0,axis=-1)
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
    #一次性多个样本进行训练来得到预测出的y值
    def get_batch(self,batch_size):
        #下标 #numpy.random.randint(low, high=None, size=None, dtype=’l’)
        #输入：low—–为最小值,high—-为最大值,size—–为数组维度大小,dtype—为数据类型，默认的数据类型是np.int。返回值：返回随机整数或整型数组，范围区间为[low,high），包含low，不包含high；high没有填写时，默认生成随机数的范围是[0，low）
        index = np.random.randint(0,self.num_train_data,batch_size)
        return self.train_data[index,:],self.train_label[index]
#模型的构建
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten层将除第一维（batch_size）以外的维度展平
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output
#模型训练参数
num_epochs = 5#循环次数
batch_size = 50#批次大小
learning_rate = 0.001#学习率
model = MLP()#模型
data_loader = MNISTloader()#数据集
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#开始训练
#批次
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
for batch_size in range(num_batches):
    x,y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(x)
        #sparse_categorical_crossentropy （交叉熵）函数，将模型的预测值 y_pred 与真实的标签值 y 作为函数参数传入，由 Keras 帮助我们计算损失函数的值
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    if batch_index % 100 == 0:                              # 每隔100个Batch保存一次
        path = checkpoint.save('./save/model.ckpt')         # 保存模型参数到文件
        print("model saved to %s" % path)
checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
checkpoint.restore(tf.train.latest_checkpoint('./save'))    # 从文件恢复模型参数
#模型的评估
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())