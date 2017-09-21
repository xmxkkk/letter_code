import tensorflow as tf
from input_data import Data
import matplotlib.pyplot as plt
import os


data=Data("./data/")


X=tf.placeholder(tf.float32,[None,40,40,3])
y=tf.placeholder(tf.float32,[None,62])
keep_prob=tf.placeholder(tf.float32)

conv1_1=tf.layers.conv2d(X,filters=6,kernel_size=4,strides=1,padding='same',activation=tf.nn.relu)
conv1_2=tf.layers.conv2d(conv1_1,filters=12,kernel_size=4,strides=1,padding='same',activation=tf.nn.relu)
pool1=tf.layers.max_pooling2d(conv1_2,pool_size=4,strides=2,padding='same')

dropout1=tf.nn.dropout(pool1,keep_prob=keep_prob)

conv2_1=tf.layers.conv2d(dropout1,filters=24,kernel_size=4,strides=1,padding='same',activation=tf.nn.relu)
conv2_2=tf.layers.conv2d(conv2_1,filters=48,kernel_size=4,strides=1,padding='same',activation=tf.nn.relu)
pool2=tf.layers.max_pooling2d(conv2_2,pool_size=4,strides=2,padding='same')
print pool2.shape
reshape1=tf.reshape(pool2,[-1,4800])
print reshape1.shape
dropout2=tf.nn.dropout(reshape1,keep_prob=keep_prob)

dense1=tf.layers.dense(dropout2,units=62)

logits=tf.nn.softmax(dense1)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense1,labels=y))
step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


plt.ion()
plt.show()

x_draw=[]
y_draw=[]

weight_path="./result/tf/weights.w"

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(),tf.initialize_local_variables()])

    if  os.path.exists(weight_path):
        saver.restore(sess, weight_path)


    train_batch_size = 100
    test_batch_size = 100

    x_axis=0

    for i in range(50*50):
        X_train, y_train = data.next_batch(train_batch_size, 'train')
        _,loss_val,accuracy_val=sess.run([step,loss,accuracy],feed_dict={X:X_train,y:y_train,keep_prob:0.7})
        print("train\tloss_val=>" + str(loss_val) + "\t\taccuracy_val=>" + str(accuracy_val))
        if i%50==49:
            X_test, y_test = data.next_batch(test_batch_size, 'test')
            loss_val,accuracy_val=sess.run([loss,accuracy],feed_dict={X:X_test,y:y_test,keep_prob:1})
            print("test\tloss_val=>"+str(loss_val)+"\t\taccuracy_val=>"+str(accuracy_val))

            x_draw.append(x_axis)
            y_draw.append(accuracy_val)
            x_axis+=1

            plt.title("tf")
            plt.plot(x_draw, y_draw, color='b')
            plt.pause(0.1)
            saver.save(sess,weight_path)

    plt.savefig("./result/tf/result.png")