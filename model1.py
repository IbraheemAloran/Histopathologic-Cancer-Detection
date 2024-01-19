import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator



#parameters
learningRate = 0.0001
batch = 64
epochs = 1
imgSize = (96,96,3)



#Dataset Paths
trainLabels = pd.read_csv('C:\\Users\\Ibraheem Aloran\\PycharmProjects\\COMP8610\\Project1\\Datasets\\train_labels.csv')
testLabels = pd.read_csv('C:\\Users\\Ibraheem Aloran\\PycharmProjects\\COMP8610\\Project1\\Datasets\\sample_submission.csv')
trainSet = "C:\\Users\\Ibraheem Aloran\\PycharmProjects\\COMP8610\\Project1\\Datasets\\train" 
testSet = "C:\\Users\\Ibraheem Aloran\\PycharmProjects\\COMP8610\\Project1\\Datasets\\test" 


testLabels['id'] = testLabels['id'] + '.tif'
trainData = trainLabels
trainData['id'] = trainData['id'] + '.tif'
trainData['label'] = trainData['label'].astype(str)

print(trainData.head())
print(testLabels.head())


#graph the correlation between negatives and positives
labelCount = trainLabels['label'].value_counts()
labelCount.plot(kind='bar',figsize=(7,5))


#Load in the datasets
idg = ImageDataGenerator(rescale=1./255, validation_split=0.2)

trainD = idg.flow_from_dataframe(
    dataframe = trainData,
    x_col='id', 
    y_col='label',
    directory=trainSet,
    subset='training',
    class_mode='binary',
    batch_size=batch,
    shuffle=True,
    target_size=(96,96))

valD=idg.flow_from_dataframe(
    dataframe=trainData,
    x_col='id', 
    y_col='label',
    directory=trainSet,
    subset="validation",
    class_mode='binary',
    batch_size=batch,
    shuffle=True,
    target_size=(96,96))

idg = ImageDataGenerator(rescale=1./255.)

idg = idg.flow_from_dataframe(
    dataframe=testLabels,
    directory=testSet,
    x_col='id', 
    y_col=None,
    target_size=(96,96),         
    batch_size=1,
    shuffle=False,
    class_mode=None)



#define the CNN and print the summary
model = keras.Sequential([
    keras.Input(shape=imgSize),


    #first convolutional block
    layers.Conv2D(16, kernel_size=(3, 3), strides=(1,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
  

    #second convolutional block
    layers.Conv2D(32, kernel_size=(3, 3), strides=(1,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
  

    #third convolutional block
    layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
   
    

    #Fully connected layer
    layers.Flatten(),
    layers.Dropout(0.2),

    #Output Layer
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    
])

model.summary()



#define the loss function and optimizer with the learning rate
optim = keras.optimizers.Adam(learning_rate=learningRate)
#lossFn = tf.keras.losses.BinaryCrossentropy() 

#Compile and Train the model
model.compile(loss='binary_crossentropy', optimizer=optim, metrics=["accuracy"])
stats = model.fit(trainD, validation_data=valD, epochs=epochs)


#Graph the losses and accuracies
plt.plot(range(epochs), stats.history["loss"])
plt.plot(range(epochs), stats.history["val_loss"])
plt.legend(["Training Loss", "Validation Loss"])
plt.title("Training and Validation Loss")
plt.show()
plt.plot(range(epochs), stats.history["accuracy"])
plt.plot(range(epochs), stats.history["val_accuracy"])
plt.legend(["Training accuracy", "Validation accuracy"])
plt.title("Training and Validation Accuracy")
plt.show()

pred = model.predict(idg, verbose=1)



#output the predicted labels to the csv file to submit on kaggle
pred = np.transpose(pred)[0]
results = pd.DataFrame()
results['id'] = testLabels['id'].apply(lambda x: x.split('.')[0])
results['label'] = list(map(lambda x: 0 if x < 0.5 else 1, pred))
results.head()



#show the bar graph correlating the negatives and positives
print(results['label'].value_counts())
results['label'].value_counts().plot(kind='bar',figsize=(7,5))



results.to_csv('submission.csv', index=False)

