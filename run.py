import tensorflow as tf
import numpy as np
from input_data import Data
from vgg16 import Vgg16

data=Data("./data224x224/")

one_image,one_label=data.next_batch(1,"train")

print(one_label.shape)

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        X=tf.placeholder("float",[None,224,224,3])
        y=tf.placeholder("float",[None,1000])

        model=Vgg16()
        model.build(X);

        cross_entropy = -tf.reduce_sum(y * tf.log(model.prob))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss=cross_entropy)

        for i in range(10000):
            X_,y_=data.next_batch(100,"train")
            sess.run(train_step,feed_dict={X:X_,y:y_})


