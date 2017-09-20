from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import os
from input_data import Data

data=Data('./data/')

model=Sequential(name="x1")

model.add(Conv2D(input_shape=(40,40,3),filters=12,kernel_size=2,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=1,padding='same'))


model.add(Dropout(0.3))

model.add(Conv2D(filters=24,kernel_size=2,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=1,padding='same'))

model.add(Flatten())

# model.add(Dense(units=200,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(units=62,activation='softmax'))


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

weight_path="./result/"+model.name+"/weights.w"


if not os.path.exists("./result/"+model.name):
    os.mkdir("./result/"+model.name)

if os.path.exists(weight_path):
    model.load_weights(weight_path)

plt.ion()
plt.show()

x_draw=[]
y_draw=[]

for i in range(10):#len(data.train_files)
    X, y = data.next_batch(100, 'train')
    model.fit(X,y,batch_size=100,epochs=1,verbose=1)
    model.save_weights(weight_path)

    X_test, y_test = data.next_batch(100, 'test')
    loss, accuracy = model.evaluate(X_test,y_test,batch_size=100)

    x_draw.append(i)
    y_draw.append(accuracy)

    plt.title(model.name)
    plt.plot(x_draw,y_draw,color='b')

    plt.pause(0.1)

plt.savefig("./result/"+model.name+"/result.png")