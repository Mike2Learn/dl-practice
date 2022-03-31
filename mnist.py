# -*- coding: utf-8 -*-

import tensorflow as tf  #导入tensorflow库
mnist=tf.keras.datasets.mnist  
(x_train,y_train),(x_test,y_test)=mnist.load_data()  #加载数据

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
y_train[0]

#数据归一化
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
x_train[0]

#搭建NN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#训练model
model.fit(x_train,y_train,epochs=6)

#测试，获取准确率
val_loss,val_acc=model.evaluate(x_test,y_test)

#进行预测
predictions=model.predict([x_test[5:8]])  #识别测试集中第6到8张图片
print(predictions)

k=0
for num in predictions[1]:
  if num==max(predictions[1]):
    break
  k=k+1
print(k)
