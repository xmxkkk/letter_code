import numpy as np
from PIL import Image
import os

files=os.listdir("./data/train");
for file in files:
    try:
        img=Image.open("./data/train/"+file)
    except:
        os.remove("./data/train/"+file)
        continue
    img=img.resize((224,224))
    img.save("./data224x224/train/"+file)