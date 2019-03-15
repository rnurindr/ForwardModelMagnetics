#!/usr/bin/env python


import numpy as np
from numpy.random import seed
np.random.seed(420)
from tensorflow import set_random_seed
import tensorflow as tf
tf.set_random_seed(420)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten,  MaxPooling2D, Conv2D
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import numpy as np
import keras.backend as K
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[45]:


#inclination predictor (please seperate to another script later, thank you)
import numpy as np
import matplotlib.pyplot as plt
i = 0
X = []
Y = []
fin = open('forward_model_with_rem_change_magnetization.csv')
for line in fin:
    arr = line.split()
    I = float(arr[0])
    D = float(arr[1])
    Im = float(arr[2])
    Dm = float(arr[3])
    if (i==0):
        num = len(arr)
        nD = int(arr[4])
    truth = True

        
    if (Im >= -90) and (Im <= -80):
        Y.append(0)
    elif (Im > -80) and (Im <= -70):
        Y.append(1)
    elif(Im > -70) and (Im <= -60):
        Y.append(2)
    elif(Im > -60) and (Im <= -50):
        Y.append(3)
    elif(Im > -50) and (Im <= -40):
        Y.append(4)
    elif(Im > -40) and (Im <= -30):
        Y.append(5)
    elif(Im > -30) and (Im <= -20):
        Y.append(6)
    elif(Im > -20) and (Im <= -10):
        Y.append(7)
    elif(Im > -10) and (Im <= 0):
        Y.append(8)
    elif(Im > 0) and (Im <= 10):
        Y.append(9)
    elif(Im > 10) and (Im <= 20):
        Y.append(10)
    elif(Im > 20) and (Im <= 30):
        Y.append(11)
    elif(Im > 30) and (Im <= 40):
        Y.append(12)
    elif(Im > 40) and (Im <= 50):
        Y.append(13)
    elif(Im > 50) and (Im <= 60):
        Y.append(14)
    elif(Im > 60) and (Im <= 70):
        Y.append(15)
    elif(Im > 70) and (Im <= 80):
        Y.append(16)
    elif(Im > 80) and (Im <= 90):
        Y.append(17)
    else:
      truth = False


    if truth:
      X.append((np.asarray(arr[5:num]).astype(np.float32)))
    if Im == 90 and Dm ==178:
        break

    
    i+=1
fin.close()

# In[47]:


# shuffle?
num = len(X)
index = np.arange(num)


np.random.shuffle(index)

numtrain = int(0.6*num)
print("numtrain:",numtrain)
numtest = int(0.2*num)
print("numtest:", numtest)
X = np.array(X)


X_shuffle = np.zeros((num, int(29*29)))
Y_shuffle = np.zeros(num)

X_train = np.zeros((numtrain, int(29*29)))
Y_train = np.zeros(numtrain)
X_valid = np.zeros((numtest, int(29*29)))
Y_train = np.zeros(numtest)

X_test = np.zeros((numtrain, int(29*29)))
Y_test = np.zeros(numtest)



for iter in range(0,num):
    X_shuffle[iter] = X[index[iter]]
    Y_shuffle[iter] = Y[index[iter]]
    

X_train = np.array(X_shuffle[:numtrain])
Y_train = np.array(Y_shuffle[:numtrain]).T
print(X_train.shape)


X_valid = np.array(X_shuffle[numtrain:numtrain+numtest])
Y_valid = np.array(Y_shuffle[numtrain:numtrain+numtest]).T
print(X_valid.shape)

X_test = np.array(X_shuffle[numtrain+numtest:numtrain+numtest+numtest])
Y_test = np.array(Y_shuffle[numtrain+numtest:numtrain+numtest+numtest]).T
print(X_test.shape)


# In[48]:


Xtrain=np.zeros((numtrain, 29,29))
Xtest = np.zeros((numtest,29,29))
Xvalid = np.zeros((numtest, 29,29))
for i in range(numtrain):
    Xtrain[i] = X_train[i].reshape(29,29)
for i in range(numtest):
    Xtest[i] = X_test[i].reshape(29,29)
    Xvalid[i] = X_valid[i].reshape(29,29)

Xtrain = np.reshape(Xtrain, (numtrain,29,29,1))
Xtest = np.reshape(Xtest, (numtest,29,29, 1))
Xvalid = np.reshape(Xvalid, (numtest,29,29, 1))
print("#train", Xtrain.shape)
print("#test", Xtest.shape)


img_rows, img_cols = 29,29

Y_test.astype(int)
Y_train.astype(int)
Y_valid.astype(int)
print(Y_train.shape)
print(Y_test.shape)


Ytest = to_categorical(Y_test)
Ytrain = to_categorical(Y_train)
Yvalid = to_categorical(Y_valid)


# In[49]:




batch_size = 273
epochs = 50


# In[50]:


test_accs = np.zeros((30,13))#.append(test_acc)
val_accs = np.zeros((30,13))#append(val_acc)
train_accs = np.zeros((30,13))#.append(train_acc)

test_losses = np.zeros((30,13))#.append(test_loss)
val_losses = np.zeros((30,13))#.append(val_loss)
train_losses = np.zeros((30,13))#.append(train_loss)


#keras.backend.clear_session()
with tf.device('/gpu:0'):
    for i in range(30):
        count = 0
        print("Iteration no.", i)
        
        #clear session before this
        keras.backend.clear_session()
        
        #store accuracy and loss values
        test_acc = []
        val_acc = []
        train_acc = []
        test_loss = []
        val_loss = []
        train_loss = []
        
        #define the models
        model0 = Sequential()
        model0.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model0.add(Activation("relu"))
        model0.add(Flatten())
        model0.add(Dense(256))
        model0.add(Activation("relu"))
        model0.add(Dense(128))
        model0.add(Activation("relu"))
        model0.add(Dense(18)) 
        model0.add(Activation(tf.nn.softmax))

        
        model1 = Sequential()
        model1.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model1.add(Activation("relu"))
        model1.add(Conv2D(32, (3, 3), padding="same"))
        model1.add(Activation("relu"))
        model1.add(Flatten())
        model1.add(Dense(256))
        model1.add(Activation("relu"))
        model1.add(Dense(128))
        model1.add(Activation("relu"))
        model1.add(Dense(18)) 
        model1.add(Activation(tf.nn.softmax))
        
        model2 = Sequential()
        model2.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model2.add(Activation("relu"))
        model2.add(Conv2D(32, (3, 3), padding="same"))
        model2.add(Activation("relu"))
        model2.add(Conv2D(32, (3, 3), padding="same"))
        model2.add(Activation("relu"))
        model2.add(Flatten())
        model2.add(Dense(256))
        model2.add(Activation("relu"))
        model2.add(Dense(128))
        model2.add(Activation("relu"))
        model2.add(Dense(18)) 
        model2.add(Activation(tf.nn.softmax))


        model3 = Sequential()
        model3.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model3.add(Activation("relu"))
        model3.add(Conv2D(32, (3, 3), padding="same"))
        model3.add(Activation("relu"))
        model3.add(Conv2D(32, (3, 3), padding="same"))
        model3.add(Activation("relu"))
        model3.add(Conv2D(64, (3, 3), padding="same"))
        model3.add(Activation("relu"))
        model3.add(Flatten())
        model3.add(Dense(256))
        model3.add(Activation("relu"))
        model3.add(Dense(128))
        model3.add(Activation("relu"))
        model3.add(Dense(18)) 
        model3.add(Activation(tf.nn.softmax))
        
        model4 = Sequential()
        model4.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model4.add(Activation("relu"))
        model4.add(Conv2D(32, (3, 3), padding="same"))
        model4.add(Activation("relu"))
        model4.add(Conv2D(32, (3, 3), padding="same"))
        model4.add(Activation("relu"))
        model4.add(Conv2D(64, (3, 3), padding="same"))
        model4.add(Activation("relu"))
        model4.add(Conv2D(64, (3, 3), padding="same"))
        model4.add(Activation("relu"))
        model4.add(Flatten())
        model4.add(Dense(256))
        model4.add(Activation("relu"))
        model4.add(Dense(128))
        model4.add(Activation("relu"))
        model4.add(Dense(18)) 
        model4.add(Activation(tf.nn.softmax))

        model5 = Sequential()
        model5.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model5.add(Activation("relu"))
        model5.add(Conv2D(32, (3, 3), padding="same"))
        model5.add(Activation("relu"))
        model5.add(Conv2D(32, (3, 3), padding="same"))
        model5.add(Activation("relu"))
        model5.add(Conv2D(64, (3, 3), padding="same"))
        model5.add(Activation("relu"))
        model5.add(Conv2D(64, (3, 3), padding="same"))
        model5.add(Activation("relu"))
        model5.add(Conv2D(64, (3, 3), padding="same"))
        model5.add(Activation("relu"))
        model5.add(Flatten())
        model5.add(Dense(256))
        model5.add(Activation("relu"))
        model5.add(Dense(128))
        model5.add(Activation("relu"))
        model5.add(Dense(18)) 
        model5.add(Activation(tf.nn.softmax))

        #3
        model6 = Sequential()
        model6.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model6.add(Activation("relu"))
        model6.add(Conv2D(32, (3, 3), padding="same"))
        model6.add(Activation("relu"))
        model6.add(Conv2D(32, (3, 3), padding="same"))
        model6.add(Activation("relu"))
        model6.add(Conv2D(64, (3, 3), padding="same"))
        model6.add(Activation("relu"))
        model6.add(Conv2D(64, (3, 3), padding="same"))
        model6.add(Activation("relu"))
        model6.add(Conv2D(64, (3, 3), padding="same"))
        model6.add(Activation("relu"))
        model6.add(Conv2D(128, (3, 3), padding="same", input_shape = (29,29,1)))
        model6.add(Activation("relu"))
        model6.add(Flatten())
        model6.add(Dense(256))
        model6.add(Activation("relu"))
        model6.add(Dense(128))
        model6.add(Activation("relu"))
        model6.add(Dense(18)) 
        model6.add(Activation(tf.nn.softmax))

        model7 = Sequential()
        model7.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model7.add(Activation("relu"))
        model7.add(Conv2D(32, (3, 3), padding="same"))
        model7.add(Activation("relu"))
        model7.add(Conv2D(32, (3, 3), padding="same"))
        model7.add(Activation("relu"))
        model7.add(Conv2D(64, (3, 3), padding="same"))
        model7.add(Activation("relu"))
        model7.add(Conv2D(64, (3, 3), padding="same"))
        model7.add(Activation("relu"))
        model7.add(Conv2D(64, (3, 3), padding="same"))
        model7.add(Activation("relu"))
        model7.add(Conv2D(128, (3, 3), padding="same", input_shape = (29,29,1)))
        model7.add(Activation("relu"))
        model7.add(Conv2D(128, (3, 3), padding="same"))
        model7.add(Activation("relu"))
        model7.add(Flatten())
        model7.add(Dense(256))
        model7.add(Activation("relu"))
        model7.add(Dense(128))
        model7.add(Activation("relu"))
        model7.add(Dense(18)) 
        model7.add(Activation(tf.nn.softmax))
        
        
        model8 = Sequential()
        model8.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model8.add(Activation("relu"))
        model8.add(Conv2D(32, (3, 3), padding="same"))
        model8.add(Activation("relu"))
        model8.add(Conv2D(32, (3, 3), padding="same"))
        model8.add(Activation("relu"))
        model8.add(Conv2D(64, (3, 3), padding="same"))
        model8.add(Activation("relu"))
        model8.add(Conv2D(64, (3, 3), padding="same"))
        model8.add(Activation("relu"))
        model8.add(Conv2D(64, (3, 3), padding="same"))
        model8.add(Activation("relu"))
        model8.add(Conv2D(128, (3, 3), padding="same", input_shape = (29,29,1)))
        model8.add(Activation("relu"))
        model8.add(Conv2D(128, (3, 3), padding="same"))
        model8.add(Activation("relu"))
        model8.add(Conv2D(128, (3, 3), padding="same"))
        model8.add(Activation("relu"))
        model8.add(Flatten())
        model8.add(Dense(256))
        model8.add(Activation("relu"))
        model8.add(Dense(128))
        model8.add(Activation("relu"))
        model8.add(Dense(18))
        model8.add(Activation(tf.nn.softmax))
        
        model9 = Sequential()
        model9.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model9.add(Activation("relu"))
        model9.add(Conv2D(32, (3, 3), padding="same"))
        model9.add(Activation("relu"))
        model9.add(Conv2D(64, (3, 3), padding="same"))
        model9.add(Activation("relu"))
        model9.add(Flatten())
        model9.add(Dense(256))
        model9.add(Activation("relu"))
        model9.add(Dense(128))
        model9.add(Activation("relu"))
        model9.add(Dense(18))
        model9.add(Activation(tf.nn.softmax))

        model10 = Sequential()
        model10.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model10.add(Activation("relu"))
        model10.add(Conv2D(32, (3, 3), padding="same"))
        model10.add(Activation("relu"))
        model10.add(Conv2D(64, (3, 3), padding="same"))
        model10.add(Activation("relu"))
        model10.add(Conv2D(64, (3, 3), padding="same"))
        model10.add(Activation("relu"))
        model10.add(Flatten())
        model10.add(Dense(256))
        model10.add(Activation("relu"))
        model10.add(Dense(128))
        model10.add(Activation("relu"))
        model10.add(Dense(18))
        model10.add(Activation(tf.nn.softmax))

        model11 = Sequential()
        model11.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model11.add(Activation("relu"))
        model11.add(Conv2D(32, (3, 3), padding="same"))
        model11.add(Activation("relu"))
        model11.add(Conv2D(64, (3, 3), padding="same"))
        model11.add(Activation("relu"))
        model11.add(Conv2D(64, (3, 3), padding="same"))
        model11.add(Activation("relu"))
        model11.add(Conv2D(128, (3, 3), padding="same"))
        model11.add(Activation("relu"))
        model11.add(Flatten())
        model11.add(Dense(256))
        model11.add(Activation("relu"))
        model11.add(Dense(128))
        model11.add(Activation("relu"))
        model11.add(Dense(18))
        model11.add(Activation(tf.nn.softmax))

        model12 = Sequential()
        model12.add(Conv2D(32, (3, 3), padding="same", input_shape = (29,29,1), kernel_initializer='random_uniform'))
        model12.add(Activation("relu"))
        model12.add(Conv2D(32, (3, 3), padding="same"))
        model12.add(Activation("relu"))
        model12.add(Conv2D(64, (3, 3), padding="same"))
        model12.add(Activation("relu"))
        model12.add(Conv2D(64, (3, 3), padding="same"))
        model12.add(Activation("relu"))
        model12.add(Conv2D(128, (3, 3), padding="same"))
        model12.add(Activation("relu"))
        model12.add(Conv2D(128, (3, 3), padding="same"))
        model12.add(Activation("relu"))
        model12.add(Flatten())
        model12.add(Dense(256))
        model12.add(Activation("relu"))
        model12.add(Dense(128))
        model12.add(Activation("relu"))
        model12.add(Dense(18))
        model12.add(Activation(tf.nn.softmax))

        models = [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10,model11, model12]

        m  = 0
        for model in models:
            #optimizer
            adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=0.0, amsgrad=False)

            #compiler/training
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam_optimizer, metrics=['accuracy'])
            history = model.fit(Xtrain, Ytrain, batch_size=batch_size,shuffle='False', epochs=epochs, verbose=0, validation_data=(Xvalid, Yvalid))
            score, acc = model.evaluate(Xtest, Ytest, batch_size=batch_size)
            print("model ", count, "done")

            #storing

            test_accs[i,m] = acc#.append(test_acc)
            val_accs[i,m] = history.history['val_acc'][-1]#append(val_acc)
            train_accs[i,m] = history.history['acc'][-1]#.append(train_acc)

            test_losses[i,m] = score#.append(test_loss)
            val_losses[i,m] = history.history['val_loss'][-1]#.append(val_loss)
            train_losses[i,m] = history.history['loss'][-1]#.append(train_loss)

            #print
            print("train_acc, train_loss",history.history['acc'][-1], history.history['loss'][-1] )
            print("val_acc, val_loss",history.history['val_acc'][-1],history.history['val_loss'][-1])
            print("test_acc, test_loss", score, acc)
            count+=1
            m += 1
        #storing
print("FINISH")
# In[173]:

np.save("InclinationTestAccuracies", test_accs)

np.save("InclinationValAccuracies", val_accs)

np.save("InclinationTrainAccuracies", train_accs)

np.save("InclinationTestLoss", test_losses)

np.save("InclinationValLoss", val_losses)

np.save("InclinationTrainLoss", train_losses)

