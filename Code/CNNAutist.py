import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import classification_report
from keras.layers import Dropout
from tensorflow.keras import datasets, layers, models

import skimage.io as io
import matplotlib.pyplot as plt

def getPlotGraphs():
  y_pred = model.predict(x_valid , verbose=1)
  y_pred_bool = np.argmax(y_pred , axis=1)
  # y_val_bool = np.argmax(y_valid , axis=1)
  y_val_bool=y_valid
  print(classification_report(y_val_bool , y_pred_bool))

  loss_train = history.history['loss']
  loss_val = history.history['val_loss']
  acc_train = history.history['accuracy']
  acc_val = history.history['val_accuracy']
  epochs = range(1 , epoch + 1)
  plt.plot(epochs , loss_train , 'g' , label='Training loss')
  plt.plot(epochs , loss_val , 'b' , label='validation loss')
  plt.plot(epochs , acc_train , 'g' , label='Training accuracy')
  plt.plot(epochs , acc_val , 'b' , label='validation accuracy')
  plt.title('Training and Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss and accuracy')
  plt.legend()
  plt.show()


def filterdata():
  WIDTH = 0
  HEIGHT = 0
  path = r'C:\Users\Dell\Desktop\CNN\autisme_detection\train\autistic'
  flag = 0
  all_images1 = []
  for image_path in os.listdir(path):
    img = io.imread(path + "\\" + image_path , as_gray=True)
    if flag == 0:
      WIDTH = img.shape[0]
      HEIGHT = img.shape[1]
      flag = 1
    img = img.reshape([WIDTH , HEIGHT , 1])
    all_images1.append(img)
  x_train1 = np.array(all_images1)
  y_train1 = np.ones(x_train1.shape[0])

  path = r'C:\Users\Dell\Desktop\CNN\autisme_detection\train\non_autistic'
  all_images2 = []
  for image_path in os.listdir(path):
    img = io.imread(path + "\\" + image_path , as_gray=True)
    if flag == 0:
      WIDTH = img.shape[0]
      HEIGHT = img.shape[1]
      flag = 1
    img = img.reshape([WIDTH , HEIGHT , 1])
    all_images2.append(img)
  x_train2 = np.array(all_images2)
  y_train2 = np.zeros(x_train2.shape[0])

  x_train = np.concatenate((x_train1 , x_train2) , axis=0)
  y_train = np.concatenate((y_train1 , y_train2) , axis=0)

  #####################################valid

  path = r'C:\Users\Dell\Desktop\CNN\autisme_detection\valid\autistic'

  all_images1 = []
  for image_path in os.listdir(path):
    img = io.imread(path + "\\" + image_path , as_gray=True)

    img = img.reshape([WIDTH , HEIGHT , 1])
    all_images1.append(img)
  x_valid1 = np.array(all_images1)
  y_valid1 = np.ones(x_valid1.shape[0])

  path = r'C:\Users\Dell\Desktop\CNN\autisme_detection\valid\non_autistic'
  all_images2 = []
  for image_path in os.listdir(path):
    img = io.imread(path + "\\" + image_path , as_gray=True)

    img = img.reshape([WIDTH , HEIGHT , 1])
    all_images2.append(img)
  x_valid2 = np.array(all_images2)
  y_valid2 = np.zeros(x_valid2.shape[0])

  x_valid = np.concatenate((x_valid1 , x_valid2) , axis=0)
  y_valid = np.concatenate((y_valid1 , y_valid2) , axis=0)

  randomize = np.arange(len(x_train))
  np.random.shuffle(randomize)
  x_train = x_train[randomize]
  y_train = y_train[randomize]

  return x_train ,y_train ,x_valid ,y_valid

x_train ,y_train ,x_valid ,y_valid= filterdata()
print(x_train.shape)
print(y_train.shape)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(2))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epoch =7
history = model.fit(x_train, y_train, epochs=epoch,batch_size=10,validation_data=(x_valid, y_valid))

getPlotGraphs()
model.save(r'C:\Users\Dell\Desktop\CNN\Model2')

test_loss, test_acc = model.evaluate(x_valid,  y_valid, verbose=2)

print(test_acc)






