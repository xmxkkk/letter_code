import tensorflow as tf
from input_data import Data

data=Data("./data/")

X=tf.placeholder(tf.float32,[None,40,40,3])
y=tf.placeholder(tf.float32,[None,62])


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# shape1=tf.reshape(X,[-1,40*40*3])
# print(shape1.shape)
# dense1=tf.layers.dense(shape1,units=40*40*6,activation=tf.nn.relu)
# print(dense1.shape)
# shape2=tf.reshape(dense1,[-1,40,40,6])

conv1_1=tf.layers.conv2d(inputs=X,filters=6,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
conv1_2=tf.layers.conv2d(inputs=conv1_1,filters=12,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
pool1=tf.layers.max_pooling2d(conv1_2,pool_size=4,strides=2,padding='same')


conv2_1=tf.layers.conv2d(inputs=pool1,filters=24,kernel_size=3,padding='same',activation=tf.nn.relu)
conv2_2=tf.layers.conv2d(inputs=conv2_1,filters=48,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
pool2=tf.layers.max_pooling2d(inputs=conv2_2,pool_size=4,strides=2,padding='same')

# conv3_1=tf.layers.conv2d(inputs=pool2,filters=96,kernel_size=3,padding='same',activation=tf.nn.relu)
# conv3_2=tf.layers.conv2d(inputs=conv3_1,filters=192,kernel_size=3,padding='same',activation=tf.nn.relu)
# pool3=tf.layers.max_pooling2d(inputs=conv3_2,pool_size=4,strides=2,padding='same')
#
# conv4_1=tf.layers.conv2d(inputs=pool3,filters=384,kernel_size=3,padding='same',activation=tf.nn.relu)
# conv4_2=tf.layers.conv2d(inputs=conv4_1,filters=768,kernel_size=3,padding='same',activation=tf.nn.relu)
# pool4=tf.layers.max_pooling2d(inputs=conv4_2,pool_size=4,strides=2,padding='same')

print(pool2.shape)

float1=tf.reshape(pool2,[-1,4800])

float=tf.layers.dense(float1,600)

pred=tf.layers.dense(float,62)

'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(pred * tf.log(y), reduction_indices=[1]))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
step=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
'''


loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=pred)           # compute cost
step = tf.train.AdamOptimizer(0.001).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(pred, axis=1),)[1]

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(),tf.initialize_local_variables()))

    for i in range(10000):
        X_,y_=data.next_batch(100,"train")
        sess.run(step,feed_dict={X:X_,y:y_})
        print("epoch=>"+str(i))
        if i%100==99:
            X_test_,y_test_=data.next_batch(100,"test")
            rr=sess.run(accuracy,feed_dict={X:X_test_,y:y_test_})

            print(rr)


