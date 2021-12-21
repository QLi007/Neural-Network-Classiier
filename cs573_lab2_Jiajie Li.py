# -*- coding: utf-8 -*-
"""
Spyder Editor
#手写数字识别
This is a temporary script file.
"""
#from _future_ import absolute_import, division, print_function, unicode_literals
#安装 TensorFlow
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import io
from sklearn.metrics  import  classification_report, confusion_matrix
#classification_report(expected_classes, predict_classes)   will give you the class accuracy and overall accuracy. 
#confusion_matrix(expected_classes, predict_classes)   will give you the confusion matrix. 
import pandas as pd
from keras.utils import to_categorical


web_url='https://archive.ics.uci.edu/ml/machine-learning-databases//'
traindata=tf.keras.utils.get_file('optdigits.tra',origin=web_url+'optdigits.tra')
testdata=tf.keras.utils.get_file('optdigits.tes',origin=web_url+'optdigits.tes')

train_data=pd.read_csv(traindata,header=None)

print(train_data)
train_y=train_data.to_numpy()[:,64]
train_data=train_data.to_numpy()[:,:64]#devide 16 denote normalization
traindata=train_data/16

test_data=pd.read_csv(testdata, header=None)
test_y=test_data.to_numpy()[:,64]
test_data=test_data.to_numpy()[:,:64]#devide 16 denote normalization
testdata=test_data/16

#   convert into 1 hot encodingoptdigits
#train_y = tf.keras.utils.to_categorical(train_y, 10)
#test_y = tf.keras.utils.to_categorical(test_y, 10)
#test_y = tf.keras.utils.to_categorical(test_y, 10).argmax(axis=1)

test_total = len(test_data)
train_total = len(train_data)
print (train_total)
print (traindata[0].shape)

# early stopping technique: stop training when no improvement in 3 consecutive epochs
callback = tf.keras.callbacks.EarlyStopping(patience=3)

#learning_rate
lr=0.06
#momentum
mt=0.01
#hidden num
hnum=30
#num of hidden units
hu=128
#input_scale=
input_scale=(64)



#build the model
test_cee_acc_hiddennum=np.zeros(hnum)
test_mse_acc_hiddennum=np.zeros(hnum)
#relu for #hidden num
for j in range (hnum):
    model1=  tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(hu, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    
    for i in range (j):
        model1.add(keras.layers.Dense(hu, activation='relu'))
    # out put layer
    #model1.add(tf.keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history1 =model1.fit(traindata, train_y, epochs=1000,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    
    print('Train Acc for crossentropy')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    print('Train confusion_matrix for crossentropy')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    print('Test Acc for crossentropy')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1)))   
    print('Test confusion_matrix for crossentropy')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))

    #hidden layer,loss='mean_squared_error',
    model2=  tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(hu, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    for i in range (j):
        model2.add(keras.layers.Dense(10, activation='relu'))
    # out put layer
    model2.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt),loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history2 =model2.fit(traindata, train_y, epochs=1000,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model2.predict(traindata)
    predict_classes_test=model2.predict(testdata)
    
    #print(model1.predict(traindata).argmax(axis=1))
    #print(model2.predict(traindata).argmax(axis=1))
    #print(model1.predict(traindata))
    #print(model2.predict(traindata))
    

    print('Train Acc for mean_squared_error')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    
    print('Train confusion_matrix for mean_squared_error')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    
    print('Test Acc for mean_squared_error')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1))) 
       
    print('Test confusion_matrix for mean_squared_error')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
    temp,test_cee_acc_hiddennum[j]=model1.evaluate(testdata,  test_y, verbose=2)
    temp,test_mse_acc_hiddennum[j]=model2.evaluate(testdata,  test_y, verbose=2)
plt.title('Test Data-RElU-CEE-hidden layer number')
#plt.title('Test Data-tanh-CEE-hidden layer number')
plt.plot(test_cee_acc_hiddennum)
plt.show()
plt.title('Test Data-RElU-MSE-hidden layer number')
plt.plot(test_mse_acc_hiddennum)
plt.show()


#D:\python3.7\lib\site-packages\sklearn\metrics\_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#  _warn_prf(average, modifier, msg_start, len(result))
#WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
#for #num of hidden units
#build the model

hu=250  
test_cee_acc_hidden_unit_num=np.zeros(hu)
test_mse_acc_hidden_unit_num=np.zeros(hu)
train_cee_acc_hidden_unit_num=np.zeros(hu)
train_mse_acc_hidden_unit_num=np.zeros(hu)
for j in range (hu):
    model1=  tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(j, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    
    for i in range (5):
#        model1.add(keras.layers.Dense(j, activation='relu'))
        model1.add(keras.layers.Dense(j, activation='tanh'))
    # out put layer
    #model1.add(tf.keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history1 =model1.fit(traindata, train_y, epochs=1000,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    
    print('Train Acc for crossentropy')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    print('Train confusion_matrix for crossentropy')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    print('Test Acc for crossentropy')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1)))   
    print('Test confusion_matrix for crossentropy')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
    #hidden layer,loss='mean_squared_error',
    model2=  tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(j, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    for i in range (30):
        model2.add(keras.layers.Dense(j, activation='relu'))
    # out put layer
    model2.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt),loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history2 =model2.fit(traindata, train_y, epochs=1000,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model2.predict(traindata)
    predict_classes_test=model2.predict(testdata)
    
    #print(model1.predict(traindata).argmax(axis=1))
    #print(model2.predict(traindata).argmax(axis=1))
    #print(model1.predict(traindata))
    #print(model2.predict(traindata))
    
    print('Train Acc for mean_squared_error')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    
    print('Train confusion_matrix for mean_squared_error')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    
    print('Test Acc for mean_squared_error')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1))) 
       
    print('Test confusion_matrix for mean_squared_error')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
    temp,test_cee_acc_hidden_unit_num[j]=model1.evaluate(testdata,  test_y, verbose=2)
    temp,test_mse_acc_hidden_unit_num[j]=model2.evaluate(testdata,  test_y, verbose=2)
plt.title('Test Data-RElU-CEE-hidden unit number')
plt.plot(test_cee_acc_hiddennum)
plt.show()
plt.title('Test Data-RElU-MSE-hidden unit number')
plt.plot(test_mse_acc_hiddennum)
plt.show()    



#for learning rate 
lr=0.01
test_cee_acc_learning_rate=np.zeros(10)
test_mse_acc_learning_rate=np.zeros(10)
train_cee_acc_learning_rate=np.zeros(10)
train_mse_acc_learning_rate=np.zeros(10)
for j in range (10):
    model1=  tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(128, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    
    for i in range (5):
        model1.add(keras.layers.Dense(128, activation='relu'))
    # out put layer
    #model1.add(tf.keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=j*0.01, momentum=mt),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history1 =model1.fit(traindata, train_y, epochs=1000,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    
    print('Train Acc for crossentropy')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    print('Train confusion_matrix for crossentropy')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    print('Test Acc for crossentropy')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1)))   
    print('Test confusion_matrix for crossentropy')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
    #hidden layer,loss='mean_squared_error',
    model2=  tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(128, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    for i in range (30):
        model2.add(keras.layers.Dense(128, activation='relu'))
    # out put layer
    model2.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=j*0.01, momentum=mt),loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history2 =model2.fit(traindata, train_y, epochs=1000,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model2.predict(traindata)
    predict_classes_test=model2.predict(testdata)
    
    #print(model1.predict(traindata).argmax(axis=1))
    #print(model2.predict(traindata).argmax(axis=1))
    #print(model1.predict(traindata))
    #print(model2.predict(traindata))
    
    print('Train Acc for mean_squared_error')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    
    print('Train confusion_matrix for mean_squared_error')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    
    print('Test Acc for mean_squared_error')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1))) 
       
    print('Test confusion_matrix for mean_squared_error')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
    temp,train_cee_acc_learning_rate[j]=model1.evaluate(traindata,  train_y, verbose=2)
    temp,train_mse_acc_learning_rate[j]=model2.evaluate(traindata,  train_y, verbose=2)
    temp,test_cee_acc_learning_rate[j]=model1.evaluate(testdata,  test_y, verbose=2)
    temp,test_mse_acc_learning_rate[j]=model2.evaluate(testdata,  test_y, verbose=2)
plt.title('Train Data-RElU-CEE-learning rate 0.01-0.1')
plt.plot(train_cee_acc_learning_rate)
plt.show()
plt.title('Train Data-RElU-MSE-learning rate 0.01-0.1')
plt.plot(train_mse_acc_learning_rate)
plt.show()  
plt.title('Test Data-RElU-CEE-learning rate 0.01-0.1')
plt.plot(test_cee_acc_learning_rate)
plt.show()
plt.title('Test Data-RElU-MSE-learning rate 0.01-0.1')
plt.plot(test_mse_acc_learning_rate)
plt.show()       

    
#for momentum 
mt=0
test_cee_acc_momentum=np.zeros(10)
test_mse_acc_momentum=np.zeros(10)
train_cee_acc_momentum=np.zeros(10)
train_mse_acc_momentum=np.zeros(10)
for j in range (10):
    model1=  tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(128, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    
    for i in range (5):
        model1.add(keras.layers.Dense(128, activation='relu'))
    # out put layer
    #model1.add(tf.keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.09, momentum=0.01*j),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history1 =model1.fit(traindata, train_y, epochs=1000,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    
    print('Train Acc for crossentropy')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    print('Train confusion_matrix for crossentropy')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    print('Test Acc for crossentropy')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1)))   
    print('Test confusion_matrix for crossentropy')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
    #hidden layer,loss='mean_squared_error',
    model2=  tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(128, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    for i in range (30):
        model2.add(keras.layers.Dense(128, activation='relu'))
    # out put layer
    model2.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.04, momentum=0.01*j),loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history2 =model2.fit(traindata, train_y, epochs=1000,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model2.predict(traindata)
    predict_classes_test=model2.predict(testdata)
    
    #print(model1.predict(traindata).argmax(axis=1))
    #print(model2.predict(traindata).argmax(axis=1))
    #print(model1.predict(traindata))
    #print(model2.predict(traindata))
    
    print('Train Acc for mean_squared_error')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    
    print('Train confusion_matrix for mean_squared_error')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    
    print('Test Acc for mean_squared_error')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1))) 
       
    print('Test confusion_matrix for mean_squared_error')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
    temp,train_cee_acc_momentum[j]=model1.evaluate(traindata,  train_y, verbose=2)
    temp,train_mse_acc_momentum[j]=model2.evaluate(traindata,  train_y, verbose=2)
    temp,test_cee_acc_momentum[j]=model1.evaluate(testdata,  test_y, verbose=2)
    temp,test_mse_acc_momentum[j]=model2.evaluate(testdata,  test_y, verbose=2)
plt.title('Train Data-RElU-CEE-momentum 0.00-0.1')
plt.plot(train_cee_acc_momentum)
plt.show()
plt.title('Train Data-RElU-MSE-momentum 0.00-0.1')
plt.plot(train_mse_acc_momentum)
plt.show()  
plt.title('Test Data-RElU-CEE-momentum 0.00-0.1')
plt.plot(test_cee_acc_momentum)
plt.show()
plt.title('Test Data-RElU-MSE-momentum 0.00-0.1')
plt.plot(test_mse_acc_momentum)
plt.show()       


#eopch step
test_cee_acc_epoch_step=np.zeros(100)
test_mse_acc_epoch_step=np.zeros(100)
train_cee_acc_epoch_step=np.zeros(100)
train_mse_acc_epoch_step=np.zeros(100)
for j in range (100):
    model1=  tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(128, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    
    for i in range (5):
        model1.add(keras.layers.Dense(128, activation='relu'))
    # out put layer
    #model1.add(tf.keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.09, momentum=0.03),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history1 =model1.fit(traindata, train_y, epochs=j*10,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    
    print('Train Acc for crossentropy')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    print('Train confusion_matrix for crossentropy')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    print('Test Acc for crossentropy')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1)))   
    print('Test confusion_matrix for crossentropy')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
    #hidden layer,loss='mean_squared_error',
    model2=  tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(128, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    for i in range (30):
        model2.add(keras.layers.Dense(128, activation='relu'))
    # out put layer
    model2.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.04, momentum=0.03),loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history2 =model2.fit(traindata, train_y, epochs=j*10,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model2.predict(traindata)
    predict_classes_test=model2.predict(testdata)
    
    #print(model1.predict(traindata).argmax(axis=1))
    #print(model2.predict(traindata).argmax(axis=1))
    #print(model1.predict(traindata))
    #print(model2.predict(traindata))
    
    print('Train Acc for mean_squared_error')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    
    print('Train confusion_matrix for mean_squared_error')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    
    print('Test Acc for mean_squared_error')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1))) 
       
    print('Test confusion_matrix for mean_squared_error')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
    temp,train_cee_acc_epoch_step[j]=model1.evaluate(traindata,  train_y, verbose=2)
    temp,train_mse_acc_epoch_step[j]=model2.evaluate(traindata,  train_y, verbose=2)
    temp,test_cee_acc_epoch_step[j]=model1.evaluate(testdata,  test_y, verbose=2)
    temp,test_mse_acc_epoch_step[j]=model2.evaluate(testdata,  test_y, verbose=2)
plt.title('Train Data-RElU-CEE-Epoch_step 10-1000')
plt.plot(train_cee_acc_epoch_step)
plt.show()
plt.title('Train Data-RElU-MSE-Epoch_step 10-1000')
plt.plot(train_mse_acc_epoch_step)
plt.show()  
plt.title('Test Data-RElU-CEE-Epoch_step 10-1000')
plt.plot(test_cee_acc_epoch_step)
plt.show()
plt.title('Test Data-RElU-MSE-Epoch_step 10-1000')
plt.plot(test_mse_acc_epoch_step)
plt.show()    


#final parameter
model1=  tf.keras.models.Sequential()
model1.add(tf.keras.layers.Flatten())
#   input layer
model1.add(keras.layers.Dense(128, input_shape=traindata[0].shape, activation='relu'))
#model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
#hidden layer,loss='categorical_crossentropy'


for i in range (5):
    model1.add(keras.layers.Dense(128, activation='relu'))
# out put layer
#model1.add(tf.keras.layers.Flatten())
model1.add(keras.layers.Dense(10, activation='softmax'))
#model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.09, momentum=0.01),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
history1 =model1.fit(traindata, train_y, epochs=50,validation_split=0.2, callbacks=[callback],verbose=0)
#model1.evaluate(traindata,  train_y, verbose=2)
#model1.evaluate(testdata,  test_y, verbose=2)
predict_classes_train=model1.predict(traindata)
predict_classes_test=model1.predict(testdata)
predict_classes_train_acc=model1.evaluate(traindata,  train_y, verbose=2)
predict_classes_test_acc=model1.evaluate(testdata,  test_y, verbose=2)

print('Train Acc for crossentropy')
print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
print('Train confusion_matrix for crossentropy')
print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
print('Test Acc for crossentropy')
print(classification_report(test_y, predict_classes_test.argmax(axis=1)))   
print('Test confusion_matrix for crossentropy')
print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))

print('train & test acc')
print(predict_classes_train_acc)
print(predict_classes_test_acc)

#hidden layer,loss='mean_squared_error',
model2=  tf.keras.models.Sequential()
model2.add(tf.keras.layers.Flatten())
#   input layer
model1.add(keras.layers.Dense(128, input_shape=traindata[0].shape, activation='relu'))
#model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
for i in range (30):
    model2.add(keras.layers.Dense(128, activation='relu'))
# out put layer
model2.add(keras.layers.Dense(10, activation='softmax'))
#model1.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.04, momentum=0.03),loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
#model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
history2 =model2.fit(traindata, train_y, epochs=380,validation_split=0.2, callbacks=[callback],verbose=0)
#model1.evaluate(traindata,  train_y, verbose=2)
#model1.evaluate(testdata,  test_y, verbose=2)
predict_classes_train=model2.predict(traindata)
predict_classes_test=model2.predict(testdata)
predict_classes_train_acc=model2.evaluate(traindata,  train_y, verbose=2)
predict_classes_test_acc=model2.evaluate(testdata,  test_y, verbose=2)
#print(model1.predict(traindata).argmax(axis=1))
#print(model2.predict(traindata).argmax(axis=1))
#print(model1.predict(traindata))
#print(model2.predict(traindata))

print('Train Acc for mean_squared_error')
print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   

print('Train confusion_matrix for mean_squared_error')
print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))

print('Test Acc for mean_squared_error')
print(classification_report(test_y, predict_classes_test.argmax(axis=1))) 
   
print('Test confusion_matrix for mean_squared_error')
print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))

print('train & test acc')
print(predict_classes_train_acc)
print(predict_classes_test_acc)



train_tanh_acc_hiddennum=np.zeros(hnum)
test_tanh_acc_hiddennum=np.zeros(hnum)
#relu for #hidden num
for j in range (hnum):
    model1=  tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(hu, input_shape=traindata[0].shape, activation='tanh'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    
    for i in range (j):
        model1.add(keras.layers.Dense(hu, activation='tanh'))
    # out put layer
    #model1.add(tf.keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history1 =model1.fit(traindata, train_y, epochs=100,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    
    print('Train Acc for crossentropy')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    print('Train confusion_matrix for crossentropy')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    print('Test Acc for crossentropy')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1)))   
    print('Test confusion_matrix for crossentropy')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
    #print(model1.predict(traindata).argmax(axis=1))
    #print(model2.predict(traindata).argmax(axis=1))
    #print(model1.predict(traindata))
    #print(model2.predict(traindata))   

    
    temp,train_tanh_acc_hiddennum[j]=model1.evaluate(traindata,  train_y, verbose=2)
    temp,test_tanh_acc_hiddennum[j]=model1.evaluate(testdata,  test_y, verbose=2)
#plt.title('Test Data-RElU-CEE-hidden layer number')
plt.title('Train Data-tanh-CEE-hidden layer number')
plt.plot(train_tanh_acc_hiddennum)
plt.show()
plt.title('Test Data-tanh-CEE-hidden layer number')
plt.plot(test_tanh_acc_hiddennum)
plt.show()



#eopch step
test_tanh_acc_epoch_step=np.zeros(50)
train_tanh_acc_epoch_step=np.zeros(50)

for j in range (50):
    model1=  tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Dense(128, input_shape=traindata[0].shape, activation='relu'))
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    
    for i in range (3):
        model1.add(keras.layers.Dense(128, activation='relu'))
    # out put layer
    #model1.add(tf.keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation='softmax'))
    #model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.09, momentum=0.03),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history1 =model1.fit(traindata, train_y, epochs=j*10,validation_split=0.2, callbacks=[callback],verbose=0)
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    
    print('Train Acc for crossentropy')
    print(classification_report(train_y, predict_classes_train.argmax(axis=1)))   
    print('Train confusion_matrix for crossentropy')
    print(confusion_matrix(train_y, predict_classes_train.argmax(axis=1)))
    print('Test Acc for crossentropy')
    print(classification_report(test_y, predict_classes_test.argmax(axis=1)))   
    print('Test confusion_matrix for crossentropy')
    print(confusion_matrix(test_y, predict_classes_test.argmax(axis=1)))
    
   
    
    temp,train_tanh_acc_epoch_step[j]=model1.evaluate(traindata,  train_y, verbose=2)

    temp,test_tanh_acc_epoch_step[j]=model1.evaluate(testdata,  test_y, verbose=2)

plt.title('Train Data-tanh-CEE-Epoch_step 10-1000')
plt.plot(train_tanh_acc_epoch_step)
plt.show()
plt.title('Test Data-tanh-CEE-Epoch_step 10-1000')
plt.plot(test_tanh_acc_epoch_step)
plt.show()


#2convolutional networks (CNNs)
#   reshape
#learning_rate
lr=0.09
#momentum
mt=0.01
#hidden num
hnum=5
#num of hidden units
hu=128

#channels num
cn=30
#rpoch stps
epnum=50

traindata = traindata.reshape(-1, 8, 8, 1)
testdata = testdata.reshape(-1, 8, 8, 1)
#   convert into 1 hot encoding
train_y = tf.keras.utils.to_categorical(train_y, 10)
test_y = tf.keras.utils.to_categorical(test_y, 10)

print(traindata[0].shape)


#channels num test
train_acc_channel_num=np.zeros(cn)
test_acc_channel_num=np.zeros(cn)
for j in range(cn):
    
    #   make model
    model1=  tf.keras.models.Sequential()
    
    #model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Conv2D(j+1, (3, 3), input_shape=traindata[0].shape, 
                                   padding='valid',activation=tf.nn.relu))
    print(j)                                 
    #   add layers
    for i in range(hnum):
        model1.add(keras.layers.Conv2D(hu, kernel_size=(2,2), activation=tf.nn.relu))
    #   output layer
    model1.add(keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    
    history = model1.fit(traindata, train_y, epochs=epnum, validation_split=0.2, verbose=2)
    
    
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    predict_classes_train_acc=model1.evaluate(traindata,  train_y, verbose=2)
    predict_classes_test_acc=model1.evaluate(testdata,  test_y, verbose=2)

    print()
    print('CNN on train data: ')
    _, train_acc = model1.evaluate(traindata, train_y)
    pred_trainy = model1.predict(traindata)
    print(classification_report(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
    print('Confusion matrix: ')
    print(confusion_matrix(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
    print('Result on test data: ')
    _, test_acc = model1.evaluate(testdata, test_y)
    pred_testy = model1.predict(testdata)
    print(classification_report(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    print('Confusion matrix: ')
    print(confusion_matrix(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    
    train_acc_channel_num[j-1]=train_acc
    test_acc_channel_num[j-1]=test_acc
plt.title('Train Data-CNN-channel num')
plt.plot(train_acc_channel_num)
plt.ylim(0.8,1)
plt.show()
plt.title('Test Data-CNN-channel num')
plt.plot(test_acc_channel_num)
plt.ylim(0.8,1)
plt.show()



#kernel size
kz=2
#kernel size test, cn=10
train_acc_kernel_size=np.zeros(kz)
test_acc_kernel_size=np.zeros(kz)
for j in range(kz):
    
    #   make model
    model1=  tf.keras.models.Sequential()
    
    #model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Conv2D(11, (3, 3), input_shape=traindata[0].shape, 
                                   padding='valid',activation=tf.nn.relu))
                                    
    #   add layers
    for i in range(hnum):
        model1.add(keras.layers.Conv2D(hu, kernel_size=(kz,kz), activation=tf.nn.relu))
    #   output layer
    model1.add(keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    
    history = model1.fit(traindata, train_y, epochs=epnum, validation_split=0.2, verbose=2)
    
    
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    predict_classes_train_acc=model1.evaluate(traindata,  train_y, verbose=2)
    predict_classes_test_acc=model1.evaluate(testdata,  test_y, verbose=2)

    print()
    print('CNN on train data: ')
    _, train_acc = model1.evaluate(traindata, train_y)
    pred_trainy = model1.predict(traindata)
    print(classification_report(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
    print('Confusion matrix: ')
    print(confusion_matrix(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
    print('Result on test data: ')
    _, test_acc = model1.evaluate(testdata, test_y)
    pred_testy = model1.predict(testdata)
    print(classification_report(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    print('Confusion matrix: ')
    print(confusion_matrix(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    
    train_acc_kernel_size[j-1]=train_acc
    test_acc_kernel_size[j-1]=test_acc
plt.title('Train Data-CNN-kernel_size')
plt.plot([1,2],train_acc_kernel_size)
plt.ylim(0.8,1)
plt.show()
plt.title('Test Data-CNN-kernel_size')
plt.plot([1,2],test_acc_kernel_size)
plt.ylim(0.8,1)
plt.show()


#epach num
epnum=100
epnum2=int(epnum/10)
#kernel size test, cn=10
train_acc_epnum=np.zeros(epnum2)
test_acc_epnum=np.zeros(epnum2)
for j in range(epnum2):
    
    #   make model
    model1=  tf.keras.models.Sequential()
    
    #model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Conv2D(11, (3, 3), input_shape=traindata[0].shape, 
                                   padding='valid',activation=tf.nn.relu))
                                    
    #   add layers
    for i in range(hnum):
        model1.add(keras.layers.Conv2D(hu, kernel_size=(1,1), activation=tf.nn.relu))
    #   output layer
    model1.add(keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
    
    history = model1.fit(traindata, train_y, epochs=(j+1)*10, validation_split=0.2, verbose=2)
    
    
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    predict_classes_train_acc=model1.evaluate(traindata,  train_y, verbose=2)
    predict_classes_test_acc=model1.evaluate(testdata,  test_y, verbose=2)

    print()
    print('CNN on train data: ')
    _, train_acc = model1.evaluate(traindata, train_y)
    pred_trainy = model1.predict(traindata)
    print(classification_report(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
    print('Confusion matrix: ')
    print(confusion_matrix(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
    print('Result on test data: ')
    _, test_acc = model1.evaluate(testdata, test_y)
    pred_testy = model1.predict(testdata)
    print(classification_report(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    print('Confusion matrix: ')
    print(confusion_matrix(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    
    train_acc_epnum[j]=train_acc
    test_acc_epnum[j]=test_acc
plt.title('Train Data-CNN-kernel_size')
plt.plot([10,20,30,40,50,60,70,80,90,100],train_acc_epnum)
plt.ylim(0.8,1)
plt.show()
plt.title('Test Data-CNN-kernel_size')
plt.plot([10,20,30,40,50,60,70,80,90,100],test_acc_epnum)
plt.ylim(0.8,1)
plt.show()



#lr,learning rate
#lr=0.1
#kernel size test, cn=10
train_acc_lr=np.zeros(10)
test_acc_lr=np.zeros(10)
for j in range(10):
    
    #   make model
    model1=  tf.keras.models.Sequential()
    
    #model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Conv2D(11, (3, 3), input_shape=traindata[0].shape, 
                                   padding='valid',activation=tf.nn.relu))
                                    
    #   add layers
    for i in range(hnum):
        model1.add(keras.layers.Conv2D(hu, kernel_size=(1,1), activation=tf.nn.relu))
    #   output layer
    model1.add(keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=j/100, momentum=mt), metrics=['accuracy'])
    
    history = model1.fit(traindata, train_y, epochs=50, validation_split=0.2, verbose=2)
    
    
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    predict_classes_train_acc=model1.evaluate(traindata,  train_y, verbose=2)
    predict_classes_test_acc=model1.evaluate(testdata,  test_y, verbose=2)

    print()
    print('CNN on train data: ')
    _, train_acc = model1.evaluate(traindata, train_y)
    pred_trainy = model1.predict(traindata)
    print(classification_report(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
    print('Confusion matrix: ')
    print(confusion_matrix(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
    print('Result on test data: ')
    _, test_acc = model1.evaluate(testdata, test_y)
    pred_testy = model1.predict(testdata)
    print(classification_report(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    print('Confusion matrix: ')
    print(confusion_matrix(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    
    train_acc_lr[j]=train_acc
    test_acc_lr[j]=test_acc
plt.title('Train Data-CNN-learning rate')
plt.plot([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],train_acc_lr)
plt.ylim(0.8,1)
plt.show()
plt.title('Test Data-CNN-learning rate')
plt.plot([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],test_acc_lr)
plt.ylim(0.8,1)
plt.show()



#momentum
#lr=0.1
#kernel size test, cn=10
train_acc_mt=np.zeros(10)
test_acc_mt=np.zeros(10)
for j in range(10):
    
    #   make model
    model1=  tf.keras.models.Sequential()
    
    #model1.add(tf.keras.layers.Flatten())
    #   input layer
    model1.add(keras.layers.Conv2D(11, (3, 3), input_shape=traindata[0].shape, 
                                   padding='valid',activation=tf.nn.relu))
                                    
    #   add layers
    for i in range(hnum):
        model1.add(keras.layers.Conv2D(hu, kernel_size=(1,1), activation=tf.nn.relu))
    #   output layer
    model1.add(keras.layers.Flatten())
    model1.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=j/100), metrics=['accuracy'])
    
    history = model1.fit(traindata, train_y, epochs=50, validation_split=0.2, verbose=2)
    
    
    #model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
    #hidden layer,loss='categorical_crossentropy'
    
    #model1.evaluate(traindata,  train_y, verbose=2)
    #model1.evaluate(testdata,  test_y, verbose=2)
    predict_classes_train=model1.predict(traindata)
    predict_classes_test=model1.predict(testdata)
    predict_classes_train_acc=model1.evaluate(traindata,  train_y, verbose=2)
    predict_classes_test_acc=model1.evaluate(testdata,  test_y, verbose=2)

    print()
    print('CNN on train data: ')
    _, train_acc = model1.evaluate(traindata, train_y)
    pred_trainy = model1.predict(traindata)
    print(classification_report(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
    print('Confusion matrix: ')
    print(confusion_matrix(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
    print('Result on test data: ')
    _, test_acc = model1.evaluate(testdata, test_y)
    pred_testy = model1.predict(testdata)
    print(classification_report(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    print('Confusion matrix: ')
    print(confusion_matrix(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    
    train_acc_mt[j]=train_acc
    test_acc_mt[j]=test_acc
plt.title('Train Data-CNN-momentum')
plt.plot([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],train_acc_mt)
plt.ylim(0.8,1)
plt.show()
plt.title('Test Data-CNN-momentum')
plt.plot([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],test_acc_mt)
plt.ylim(0.8,1)
plt.show()


#momentum
#lr=0.1


    
#   make model
model1=  tf.keras.models.Sequential()

#model1.add(tf.keras.layers.Flatten())
#   input layer
model1.add(keras.layers.Conv2D(11, (3, 3), input_shape=traindata[0].shape, 
                               padding='valid',activation=tf.nn.relu))
                                
#   add layers
for i in range(hnum):
    model1.add(keras.layers.Conv2D(hu, kernel_size=(1,1), activation=tf.nn.relu))
#   output layer
model1.add(keras.layers.Flatten())
model1.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model1.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0), metrics=['accuracy'])

history = model1.fit(traindata, train_y, epochs=50, validation_split=0.2, verbose=2)


#model1.add(keras.layers.Dense(hu, input_shape=(1,64), activation='relu'))
#hidden layer,loss='categorical_crossentropy'

#model1.evaluate(traindata,  train_y, verbose=2)
#model1.evaluate(testdata,  test_y, verbose=2)
predict_classes_train=model1.predict(traindata)
predict_classes_test=model1.predict(testdata)
predict_classes_train_acc=model1.evaluate(traindata,  train_y, verbose=2)
predict_classes_test_acc=model1.evaluate(testdata,  test_y, verbose=2)

print()
print('CNN on train data: ')
_, train_acc = model1.evaluate(traindata, train_y)
pred_trainy = model1.predict(traindata)
print(classification_report(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
print('Confusion matrix: ')
print(confusion_matrix(train_y.argmax(axis=1), pred_trainy.argmax(axis=1)))
print('Result on test data: ')
_, test_acc = model1.evaluate(testdata, test_y)
pred_testy = model1.predict(testdata)
print(classification_report(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
print('Confusion matrix: ')
print(confusion_matrix(test_y.argmax(axis=1), pred_testy.argmax(axis=1)))
    
