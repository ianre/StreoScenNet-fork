
#! export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
from __future__ import print_function
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
from data import load_train_data, load_test_data,plot_imagesT,destroy_train_test
import pdb
from skimage.io import imsave, imread
import cv2
import pickle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import data
import pylab
import imageio
import matplotlib.pyplot as plt
#from  gen_data import load_image,random_batch,test_batch,load_images
from  get_resUnet import *
import params 
from os.path import splitext
from keras.utils import plot_model
from pathlib import Path
#import dask.array as da
#from dask.delayed import delayed
from dataloaders import *
print("finished imports")

netparam=params.init() 
netprameval=params.init(train=0) 

res = data.destroy_train_test()
print(res)

data.create_train_data(netparam)
print("------------\nfinished create_train_data\n------------")
data.create_test_data(netparam)
print("netparam",netparam)

#netparam.task

imgs_train, imgs_mask_train=data.load_train_data()
imgs_test,  imgs_mask_test =data.load_test_data()

train_setT=np.shape(imgs_train)
print("imgs_train, imgs_mask_train",len(imgs_train), len(imgs_mask_train))
print("imgs_test,  imgs_mask_test",len(imgs_test), len(imgs_mask_test))

#imgs_test[27]
#imgs_mask_test[27]

np.random.seed(1234)
ids_train = np.random.choice(len(imgs_train), size=int(len(imgs_train)/1), replace=False)
print("ids_train",ids_train,"from:len(imgs_train)",len(imgs_train),"is",len(ids_train),"length")
ids_val = np.random.choice(len(imgs_test), size=int(len(imgs_test)/1), replace=False)
print("ids_val",ids_val,"from:len(imgs_test)",len(imgs_test),"is",len(ids_val),"length")

x=[[1,3],[2,5],[4,8],[6,7]]
d1=str(x[3][0])
d2=str(x[3][1])

print("Data Information: Evaluation on %s and %s"% (d1,d2))
print("------")
print("  - No of Frames in Training set: %d" % len(ids_train))
print("  - No of Frames in Test set %d" % len(ids_val))

datagen = CustomImageDataGenerator(netparam, training =0)

trainflow=datagen.flow(imgs_train, imgs_mask_train, batch_size=8)

x_batch,x_batch_right, y_batch,_ = trainflow.next()

y_batch.shape

#x_batch, y_batch,_ = trainflow.next()
#plot_imagesT((x_batch[:,:,:,0:3]*255).astype(np.uint8),(x_batch_right[:,:,:,0:3]*255).astype(np.uint8), np.squeeze(y_batch*255), cls_pred=None, smooth=True, filename='test.png')

y_batch.shape
imgs_test=[imgs_test[img] for img in ids_val]
imgs_mask_test=[imgs_mask_test[img] for img in ids_val]

#ids_val_batch = np.random.choice(len(imgs_test), size=val_num, replace=False)
imgs_test=np.array(imgs_test)
imgs_mask_test=np.array(imgs_mask_test)
print("imgs_test:imgs_mask_test",len(imgs_test),len(imgs_mask_test))
#imgs_mask_test =imgs_mask_test[ids_val_batch]

imgs_train=[imgs_train[img] for img in ids_train]
imgs_mask_train=[imgs_mask_train[img] for img in ids_train]


imgs_train=np.array(imgs_train)
imgs_mask_train=np.array(imgs_mask_train)
print("imgs_train:imgs_mask_train",len(imgs_train),len(imgs_mask_train))


trainflow=datagen.flow(imgs_train, imgs_mask_train, batch_size=1)
x_batch,x_batch_right, y_batch,_ = trainflow.next()


y_batch.shape

batchsize=10
 
Traindatagen =CustomImageDataGenerator(netparam,training=1)

Validdatagen= CustomImageDataGenerator(netprameval,training=1)

trainflow=Traindatagen.flow(imgs_train, imgs_mask_train, batch_size=1)
x_batch,x_batch_right, y_batch,_ = trainflow.next()

y_batch.shape

def train_generator():
    trainflow=Traindatagen.flow(imgs_train, imgs_mask_train, batch_size=batchsize)
    while True:
        x_batch,x_batch_right, y_batch,_ = trainflow.next()
        #pdb.set_trace()
        yield [x_batch,x_batch_right], y_batch
print("finished train_generator")

def valid_generator():
    validflow=Validdatagen.flow(imgs_test, imgs_mask_test, batch_size=batchsize)
    while True:
        x_batch,x_batch_right, y_batch,_ = validflow.next()
#        pdb.set_trace()
        yield [x_batch, x_batch_right],y_batch



model =YnetResNet(netparam)
filename='YnetResNet2017_'+netparam.task+'_v'+d1+d2+'.hdf5'
logdirs='%s%s'%(splitext(filename)[0],'logs')
print("finished get YnetResNet")

if  (os.path.exists(filename)):
    print("Loading model from disk")
    model.load_weights(filename, by_name=False)

model.summary()

plot_model(model, to_file='%s%s'%(splitext(filename)[0],'model2017.png'))


tensorboard = TensorBoard(log_dir=logdirs)
callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=7,
                               verbose=1,
                               min_delta=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath=filename,
                             save_best_only=True,
                             save_weights_only=True),
             tensorboard]



history =model.fit(train_generator(),
                    steps_per_epoch=np.ceil(float(len(imgs_train)) / float(netparam.batch_size)),
                    epochs=50,
                    use_multiprocessing=False,
                    max_queue_size=50, 
                    workers=1,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(imgs_test)) / float(netparam.batch_size)))


#histfile='%s%s'%(splitext(filename)[0],'_hist')
histfile = "/home/student/Documents/GitHub/StreoScenNet-fork/models/hist/hist50"
with open(histfile, 'wb') as f: 
    pickle.dump([history.history], f)


#history.save_model('/home/student/Documents/GitHub/StreoScenNet-fork/models/test_model')


fr_num=20
'''
img_left=imread(imgs_test[fr_num][0])
img_right=imread(imgs_test[fr_num][1])
'''
img_left=imread(imgs_train[fr_num][0])
img_right=imread(imgs_train[fr_num][1])

img_left = img_left.astype('float32')
img_right = img_right.astype('float32')
img_left/=255.
img_right/=255.
img_left =cv2.resize(img_left, (224,224))
img_right =cv2.resize(img_right, (224,224))
img_right=np.reshape(img_right,(-1,img_left.shape[0],img_left.shape[1],img_left.shape[2]))
img_left=np.reshape(img_left,(-1,img_left.shape[0],img_left.shape[1],img_left.shape[2]))
pred_y_batch = model.predict([img_left, img_right], batch_size=4,verbose=1)
with open('labels_2017.json') as json_file:
    dataf = json.load(json_file)

def convert_color(data,im, tasktype):
   # pdb.set_trace()
    im=np.squeeze(im)
    if tasktype.task=='all':
        out1 = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)
        out2 = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)
        out3 = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)
        for label_info,index in zip(data['instrument'],range(0,np.shape(data['instrument'])[0]+1)):
            rgb=label_info['color'][0]
            if np.sum(rgb)==0:
                continue
            temp=im[:,:,index-1]
            temp=temp.astype(np.float)
            #temp =cv2.resize(temp,(224,224),interpolation=cv2.INTER_CUBIC)
            match_pxls = np.where(temp > 0.2)
            out1[match_pxls] = rgb
            
        for label_info,index in zip(data['parts'],range(np.shape(data['instrument'])[0],np.shape(data['instrument'])[0]+np.shape(data['parts'])[0])):
            rgb=label_info['color'][1]
            #pdb.set_trace()
            if np.sum(rgb)==0:
                continue
            temp=im[:,:,index-1]
            #print(index-1)
            temp=temp.astype(np.float)
            #temp =cv2.resize(temp,(224,224),interpolation=cv2.INTER_CUBIC)
            match_pxls = np.where(temp > 0.2)
            out2[match_pxls] = rgb
        out3=(im[:,:,index]>0.2)*255
        out=np.dstack((out1,out2,out3))
    if tasktype.task=='binary':
        out = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)  
        out=(im>0.2)*255
    if tasktype.task=='parts':
        out = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)
        for label_info,index in zip(data['parts'],range(0,np.shape(data['parts'])[0])):
            rgb=label_info['color'][1]
            if np.sum(rgb)==0:
                continue
            temp=im[:,:,index]
            temp=temp.astype(np.float)
            temp =cv2.resize(temp,(224,224),interpolation=cv2.INTER_CUBIC)
            match_pxls = np.where(temp > 0.2)
            out[match_pxls] = rgb
    if tasktype.task=='instrument':
        out = (np.zeros((im.shape[0],im.shape[1])) ).astype(np.uint8)
        for label_info,index in zip(data['instrument'],range(0,np.shape(data['instrument'])[0])):
            rgb=label_info['color'][0]
            if np.sum(rgb)==0:
                continue
            temp=im[:,:,index-1]
            temp=temp.astype(np.float)
            temp =cv2.resize(temp,(224,224),interpolation=cv2.INTER_CUBIC)
            match_pxls = np.where(temp > 0.2)
            out[match_pxls] = rgb
    return out.astype(np.uint8)

def mask_color_img(img,parts,binary, instr,fname='tr'):
    parts=np.squeeze(parts)
    binary=np.squeeze(binary)
    instr=np.squeeze(instr)
    T=[[127,201,127],
    [190,174,212],
    [253,192,134],
    [255,255,153],
    [56, 108,176],
    [240,  2,127],
    [191, 91,23]]
    P=[[255,0,0],
    [0,255,0],
    [0,0,255]]
    gt_b=np.dstack((np.zeros((224, 224)).astype(np.uint8),np.zeros((224, 224)).astype(np.uint8),binary)).astype(np.uint8)
    draw_img_b = cv2.addWeighted(img,1,gt_b,0.8,0.5).astype(np.uint8)
    gt_inst= (np.zeros((img.shape[0],img.shape[1],3)) ).astype(np.uint8)
    for g in range(1,8):
        match_pxls = np.where(instr == (g)*32)
        gt_inst[match_pxls] = T[g-1]
        #print(T[g])
    draw_img_inst = cv2.addWeighted(img,1,gt_inst,0.5,0.5).astype(np.uint8)
    gt_part= (np.zeros((img.shape[0],img.shape[1],3)) ).astype(np.uint8)
    for g in range(1,4):
        match_pxls = np.where(parts == (g)*85)
        gt_part[match_pxls] = P[g-1]
        #print(T[g])
    draw_img_part = cv2.addWeighted(img,1,gt_part,1,0.5).astype(np.uint8)
    
    cv2.imwrite(fname+'_binary.png', gt_b[...,::-1])
    plt.imshow(draw_img_b)
    plt.show()
    cv2.imwrite(fname+'_inst.png', gt_inst[...,::-1])
    plt.imshow(draw_img_inst)
    plt.show()
    cv2.imwrite(fname+'_part.png', gt_part[...,::-1])
    plt.imshow(draw_img_part)
    plt.show()
    return

pred_y_batch=np.squeeze(pred_y_batch)
temp=convert_color(dataf,(pred_y_batch), netparam)


#par='_'+(imgs_train[fr_num][0]).split('/')[-3][-1]+'_'+((imgs_train[fr_num][0]).split('/')[-1][:-4])
par='_'+(imgs_test[26][0]).split('/')[-3][-1]+'_'+((imgs_test[26][0]).split('/')[-1][:-4])


img=imread(imgs_test[26][0])
mask_color_img(cv2.resize(img, (224,224)), temp[:,:,1],temp[:,:,2], temp[:,:,0],fname='pred'+par)

