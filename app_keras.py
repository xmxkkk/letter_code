# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Reshape
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import os
from input_data import Data
from keras.optimizers import Adam

data=Data('./data/')

def build1(_name):
    model=Sequential(name=_name)
    model.add(Conv2D(input_shape=(40,40,3),filters=12,kernel_size=2,strides=1,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=1,padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=24,kernel_size=2,strides=1,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=1,padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(units=62,activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build2(_name):
    model=Sequential(name=_name)

    model.add(Conv2D(input_shape=(40,40,3),filters=6,kernel_size=2,strides=1,padding='same',activation='relu'))
    model.add(Conv2D(filters=12,kernel_size=2,strides=1,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=4,strides=2,padding='same'))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=24, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(units=62,activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build3(_name):
    model = Sequential(name=_name)

    model.add(Conv2D(input_shape=(40, 40, 3), filters=6, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=12, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=24, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(units=62, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build4(_name):
    model = Sequential(name=_name)

    model.add(Conv2D(input_shape=(40, 40, 3), filters=6, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=12, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=24, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(units=62, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build5(_name):
    model = Sequential(name=_name)

    model.add(Conv2D(input_shape=(40, 40, 3), filters=6, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=12, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=5, strides=2, padding='same'))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=24, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=5, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(units=62, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build6(_name):
    model = Sequential(name=_name)

    model.add(Conv2D(input_shape=(40, 40, 3), filters=12, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=24, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=48, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=96, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(units=62, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build7(_name):
    model = Sequential(name=_name)

    model.add(Conv2D(input_shape=(40, 40, 3), filters=6, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=12, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=24, kernel_size=2, strides=1, padding='same', activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(units=62, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build8(_name):
    model = Sequential(name=_name)

    model.add(Conv2D(input_shape=(40, 40, 3), filters=6, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=12, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=24, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(units=62, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build9(_name):
    model = Sequential(name=_name)

    model.add(Conv2D(input_shape=(40, 40, 3), filters=6, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=12, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=24, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Flatten())

    model.add(Dropout(0.3))

    model.add(Dense(units=256,activation='relu'))

    model.add(Dense(units=62, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build10(_name):
    model = Sequential(name=_name)

    model.add(Conv2D(input_shape=(40, 40, 3), filters=12, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))
    model.add(Conv2D(filters=24, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=48, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))
    model.add(Conv2D(filters=96, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(units=62, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build11(_name):
    model = Sequential(name=_name)

    model.add(Conv2D(input_shape=(40, 40, 3), filters=12, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=24, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=48, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(units=62, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build12(_name):
    model = Sequential(name=_name)

    model.add(Conv2D(input_shape=(40, 40, 3), filters=6, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=12, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=24, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=48, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(units=62, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# model=build1("x1")
# model=build2("x2")
# model=build3("x3") #收敛快
# model=build4("x4")
# model=build5("x5")
# model=build6("x6")
# model=build7("x7")
# model=build8("x8")
# model=build9("x9")
# model=build10("x10")
# model=build11("x11")

model=build12("x12")

print(model.summary())

weight_path="./result/"+model.name+"/weights.w"
if not os.path.exists("./result/"+model.name):
    os.mkdir("./result/"+model.name)

if os.path.exists(weight_path):
    model.load_weights(weight_path)

plt.ion()
plt.show()

x_draw=[]
y_draw=[]

train_batch_size=5000
test_batch_size=100

for i in range(50):
    X, y = data.next_batch(train_batch_size, 'train')
    model.fit(X,y,batch_size=100,epochs=1,verbose=1)
    model.save_weights(weight_path)


    X_test, y_test = data.next_batch(test_batch_size, 'test')
    loss, accuracy = model.evaluate(X_test,y_test,batch_size=test_batch_size)

    print("epoch"+str(i)+",accuracy=>"+str(accuracy))

    x_draw.append(i)
    y_draw.append(accuracy)

    plt.title(model.name)
    plt.plot(x_draw,y_draw,color='b')

    plt.pause(0.1)

plt.savefig("./result/"+model.name+"/result.png")
''''''