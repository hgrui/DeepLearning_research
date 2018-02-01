from __future__ import print_function
import os
import keras
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Activation,Input,Concatenate,Dropout,GlobalAveragePooling2D
from keras.models import Model
import time
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.utils import get_file
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#SqueezeNet

#backend=tf  channels_last (rows,cols,channels)


 
def fire_module(input,squeeze_filters,expand_filters):
    squeeze=Conv2D(squeeze_filters,
                   kernel_size=(1,1),
                   strides=1,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   data_format='channels_last'
                   )(input)
    relu_squeeze=Activation('relu')(squeeze)
    expand1=Conv2D(expand_filters,
                   kernel_size=(1,1),
                   strides=1,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   data_format='channels_last'
                   )(relu_squeeze)
    relu_expand1=Activation('relu')(expand1)
    expand2=Conv2D(expand_filters,
                   kernel_size=(3,3),
                   strides=1,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   data_format='channels_last'
                   )(relu_squeeze)
    relu_expand2=Activation('relu')(expand2)
    merge=Concatenate(axis=3)([relu_expand1,relu_expand2])
    output=merge
    return output

def SqueezeNet(input_shape,num_classes,weight=None):
    input=Input(shape=input_shape)
    conv_1=Conv2D(96,
                  kernel_size=(7,7),
                  strides=2,
                  padding='same',
                  kernel_initializer='glorot_uniform'
                  )(input)
    pool_1=MaxPooling2D(pool_size=(3,3),
                        strides=2)(conv_1)
    fire_2=fire_module(pool_1,16,64)
    fire_3=fire_module(fire_2,16,64)
    fire_4=fire_module(fire_3,32,128)
    pool_4=MaxPooling2D(pool_size=(3,3),
                        strides=2)(fire_4)
    fire_5=fire_module(pool_4,32,128)
    fire_6=fire_module(fire_5,48,192)
    fire_7=fire_module(fire_6,48,192)
    fire_8=fire_module(fire_7,64,256)
    pool_8=MaxPooling2D(pool_size=(3,3),
                        strides=2)(fire_8)
    fire_9=fire_module(pool_8,64,256)
    drop=Dropout(0.5)(fire_9)
    conv_10=Conv2D(num_classes,
                   kernel_size=(1,1),
                   strides=1,
                   padding='same',
                   kernel_initializer='glorot_uniform'
                   )(drop)
    relu_11=Activation('relu')(conv_10)
    avgpool=GlobalAveragePooling2D()(relu_11)


    flatten=Flatten()(relu_11)
    dense=Dense(64)(flatten)
    relu_dense=Activation('relu')(dense)
    dense=Dense(2)(relu_dense)
    softmax1=Activation('softmax')(dense)

    softmax=Activation('softmax')(avgpool)
    print(softmax)
    output=softmax
    model=Model(input=input,output=output)
 
    return model

    




def main():
    t0=time.time()
    batch_size = 32
    num_classes = 10
    epochs = 20
    data_augmentation = True
    print('start')
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape)

    x_train_n = np.zeros((x_train.shape[0], 224, 224, 3),dtype = 'float16')
    x_test_n = np.zeros((x_test.shape[0], 224, 224, 3),dtype = 'float16')

    for i in range(x_train.shape[0]):
        if i%5000==0:
            print(i)
        data=x_train[i]
        img=image.array_to_img(data)
        img2=img.resize((224,224))
        data2=image.img_to_array(img2)
        x_train_n[i,:]=data2
    for i in range(x_test.shape[0]):
        if i%2000==0:
            print(i)
        data=x_test[i]
        img=image.array_to_img(data)
        img2=img.resize((224,224))
        data2=image.img_to_array(img2)
        x_test_n[i,:]=data2

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train_n /= 255.0
    x_test_n /= 255.0

    model=SqueezeNet((224,224,3),10)
    model.summary()
    print('wow')
    print(time.time()-t0)
    sgd=SGD(lr=0.01,decay=0.0002,momentum=0.9)
    model.compile(optimizer=sgd,                  
                  loss='categorical_crossentropy',                  
                  metrics=['accuracy'])
    model.fit(x_train_n, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test_n, y_test),
            shuffle=True)


if __name__=='__main__':
    main()