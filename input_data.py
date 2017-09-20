import numpy as np
from PIL import Image
import os
import random

class Data:
    def __init__(self,path):
        self.path=path
        self.train_files=os.listdir(path+"train")
        self.test_files=os.listdir(path+"test")
        self.train_idx=0
        self.test_idx=0
        self.labels="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def next_batch(self,batch_size=100,type="train"):
        idx=0
        files=None
        rotate=0
        if type=="train":
            idx=self.train_idx
            files=self.train_files
            rotate=random.randint(-30,30)
            print("rotate:"+str(rotate))
        else:
            idx=self.test_idx
            files=self.test_files



        if idx*batch_size+batch_size>len(files):
            idx=0

        datas=files[idx*batch_size:idx*batch_size+batch_size]
        X=[]
        y=[]
        for data in datas:
            img=Image.open(self.path+type+"/"+data)
            img=img.convert("RGB")
            img.rotate(rotate)
            img=np.array(img,np.float32)/256.0

            X.append(img)

            letter=data[-5:-4]


            label=[0]*len(self.labels)
            label[self.labels.index(letter)]=1
            y.append(label)



        if type=="train":
            self.train_idx+=1
        else:
            self.test_idx+=1

        return np.array(X),np.array(y)

# data=Data("./data/")
# X,y=data.next_batch(1,"train")
# print(X.shape,y.shape)
# print(X)
# print(y)