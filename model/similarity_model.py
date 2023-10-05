import numpy as np
import pandas as pd
import gc
from time import time
from tqdm import tqdm_notebook,trange,tqdm
import argparse
import os,sys
import heapq
import io
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Input, dot

import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath('../keras-yolo3-detection/')
sys.path.append(ROOT_DIR)
from yolo import YOLO, detect_video
from PIL import Image

def argsm():
    parser = argparse.ArgumentParser(description="Run similarity model.")
    parser.add_argument('--path', nargs='?', default='../img/1.jpg',
                        help='Input data path.')

    return parser.parse_args()

def feature_extractor(path):
    # p is image feature
    
    img = Image.open(path)
    r_image,a,b,c,d = YOLO().detect_image(img)
    crop_image=r_image.crop((a,b,c,d))
    new_image=crop_image.resize((224, 224))
    x = image.img_to_array(new_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    p = model.predict(x)[0][0][0]

    return p,r_image,crop_image

def similarity_model():
    item1 = Input(shape=(2048,))
    item2 = Input(shape=(2048,))

    x = dot([item1,item2],axes=1,normalize=True)# normalize 하자..

    return Model(inputs=[item1,item2], outputs=x)

def rank_figure(byte_list):
    f, (ax0, ax1, ax2, ax3, ax4)=plt.subplots(1, 5, figsize=(25, 5))
    ax0.imshow(r_image)
    ax0.set_axis_off()
    ax0.set_title('query')

    ax1.imshow(crop_image)
    ax1.set_axis_off()
    ax1.set_title('detect fashion')
    
    ax2.imshow(Image.open(io.BytesIO(byte_list[0])))
    ax2.set_axis_off()
    ax2.set_title('top-1')
    
    ax3.imshow(Image.open(io.BytesIO(byte_list[1])))
    ax3.set_axis_off()
    ax3.set_title('top-2')
    
    ax4.imshow(Image.open(io.BytesIO(byte_list[2])))
    ax4.set_axis_off()
    ax4.set_title('top-3')
    
    plt.show()
    
    return 
    
if __name__ == '__main__':
    args = argsm()
    path = args.path


t1=time()
dataset = np.load('../dataset/AmazonFashion6ImgPartitioned.npy',encoding='bytes',allow_pickle=True)
train, val, test, meta, usernum, itemnum = dataset
print('data load complete %.2fs'%(time()-t1))

#######################################
# start detect
#######################################  
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
print('model load complete')


######################
# image features load
######################
file = '../dataset/amazonfashion6_imgfeature.hdf5'
img_data = HDF5Matrix(file,'imgs')
print('image feature load complete')

##################
# extract feature
##################  
p,r_image,crop_image=feature_extractor(path)

#######################################
# similar product search
#######################################
print('find similar product')

a=[]
b=[]
for product in img_data:
    a.append(product)
    b.append(p)
sim_model = similarity_model()
pp=sim_model.predict([a,b],batch_size=2**15)
item_score = {n: pp[n] for n, i in enumerate(img_data)}
ranklist = heapq.nlargest(10, item_score, key=item_score.get)



byte_list = [meta[i][b'imgs'] for i in ranklist]
rank_figure(byte_list)
    
