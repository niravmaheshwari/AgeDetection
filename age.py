#importing libraries
import os

import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense



# Initialising the model
regressor = Sequential()


regressor.add(Conv2D(64, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))


regressor.add(MaxPooling2D(pool_size = (2, 2)))


regressor.add(Conv2D(64, (3, 3), activation = 'relu'))


regressor.add(MaxPooling2D(pool_size = (2, 2)))



regressor.add(Conv2D(64, (3, 3), activation = 'relu'))


regressor.add(MaxPooling2D(pool_size = (2, 2)))


regressor.add(Flatten())


regressor.add(Dense(units = 256, activation = 'relu'))


regressor.add(Dense(units = 256, activation = 'relu'))


regressor.add(Dense(units = 1, activation = 'linear'))


#compiling the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

#making training data from input images


from PIL import Image

y_train=[]

for root, dirs, files in os.walk("."): 
    i=0
    for filename in files:
        if i==0:
            img = Image.open(filename)
            i+=1
            img = img.resize((128,128), Image.ANTIALIAS)
            array=np.array((img))
            array=np.expand_dims(array,axis=0)
            X_train=array
            left_text = filename.partition("_")[0]
            y_train.append(int(left_text))

        else:
            img = Image.open(filename)
            img = img.resize((128,128), Image.ANTIALIAS)
            array = np.array(img)
            if array.ndim==3:
                if np.shape(array)[2]==3:
                    array=np.expand_dims(array,axis=0)
                    X_train=np.append(X_train,array,axis=0)
                    left_text = filename.partition("_")[0]
                    y_train.append(int(left_text))
        

y_train=np.array(y_train)




y_test=[]

for root, dirs, files in os.walk("."): 
    i=0
    for filename in files:
        if i==0:
            img = Image.open(filename)
            i+=1
            img = img.resize((128,128), Image.ANTIALIAS)
            array=np.array((img))
            array=np.expand_dims(array,axis=0)
            X_test=array
            left_text = filename.partition("_")[0]
            y_test.append(int(left_text))

        else:
            img = Image.open(filename)
            img = img.resize((128,128), Image.ANTIALIAS)
            array = np.array(img)
            if array.ndim==3:
                if np.shape(array)[2]==3:
                    array=np.expand_dims(array,axis=0)
                    X_test=np.append(X_test,array,axis=0)
                    left_text = filename.partition("_")[0]
                    y_test.append(int(left_text))

y_test=np.array(y_test)


#fitting the model to training data
regressor.fit(X_train, y_train, batch_size=32, nb_epoch=50,shuffle=True)

y_pred=regressor.predict(X_test)
