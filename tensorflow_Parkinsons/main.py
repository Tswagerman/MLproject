import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
#from sklearn.model_selection import train_test_split

class superVisedLearner:
    def __init__(self, dataLocation):  
        self.__createDatasets(dataLocation)

    def __createDatasets(self, dataLocation):
        parkinsons_dataset = pd.read_csv(dataLocation, names=["name","MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)",
                                                                   "MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3",
                                                                   "Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","status","RPDE","DFA","spread1",
                                                                   "spread2","D2","PPE"], skiprows=[0]) #Skipping first row in csv file
        #The string variable 'name' will cause errors when converting to np.array. 
        #Also, it is not relevant in order to predict whether an individual has parkinson's disease.
        parkinsons_dataset.drop("name", inplace=True, axis=1) 
        #The aim was to divide the training and testing data 80/20.
        #In order to divide the test and train sets in between subjects,
        #the percentage had to be quite precise. This resulted in an arbitrary 0.785 percent of the data.
        parkinsons_training_dataset = self.__takeSampleRows(parkinsons_dataset, 0.785) 
        print(parkinsons_training_dataset)
        parkinsons_training_features = parkinsons_training_dataset.copy()
        parkinsons_training_labels = parkinsons_training_features.pop('status') #Health status of the subject (one) - Parkinson's, (zero) - healthy
        parkinsons_training_features = np.array(parkinsons_training_features)
        parkinsons_training_labels = np.array(parkinsons_training_labels)

        #The remainder of the data is used for testing the model.
        parkinsons_testing_dataset = parkinsons_dataset.iloc[max(parkinsons_training_dataset.index):]
        parkinsons_testing_features = parkinsons_testing_dataset.copy()
        parkinsons_testing_labels = parkinsons_testing_features.pop('status')
        parkinsons_testing_features = np.array(parkinsons_testing_features)
        parkinsons_testing_labels = np.array(parkinsons_testing_labels)
        #The model is created based on the training dataset.
        model = self.__createModel(parkinsons_training_features, parkinsons_training_labels)
        #The testing data is used to validate the accuracy of the model.
        self.__make_predictions(model, parkinsons_testing_features, parkinsons_testing_labels, "Testing Data ")
   

    def __createModel(self, parkinsons_training_features, parkinsons_training_labels):
        features_shape = parkinsons_training_features.shape[1:]
        label_variability = 2 #Either healthy or not
        # Create a Normalization layer and set its internal state using the training data
        normalizer = preprocessing.Normalization()
        normalizer.adapt(parkinsons_training_features)

        #Create model layers
        input = keras.Input(shape=features_shape)
        features = normalizer(input)
        output = layers.Dense(label_variability, activation="softmax")(features)
        model = keras.Model(input, output)

        #Train model
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit(parkinsons_training_features, parkinsons_training_labels, epochs=10)

        self.__make_predictions(model, parkinsons_training_features, parkinsons_training_labels, "Training Data")
        return model


    def __takeSampleRows(self, data, perc):
        return data.head(int(len(data)*(perc)))

    def __make_predictions(self, model, parkinsons_features, parkinsons_labels, string):
        print("\n############   ", string, "  ############")
        predictions = model(parkinsons_features)
        #print(predictions)
        tf.nn.softmax(predictions)
        predictionArray = tf.argmax(predictions, axis=1)
        self.__calc_accuracy(np.array(predictionArray), parkinsons_labels)
        
    def __calc_accuracy(self, predictionArray, labelArray):
        accurate_counter = 0
        #print("PREDICTION ARRAY OF HEALTH STATUS:\n", predictionArray)
        #print("ACTUAL HEALTH STATUS\n", labelArray)
        for iteration in range(labelArray.size):
            if (predictionArray[iteration] == labelArray[iteration]):
                accurate_counter += 1
        accuracy = (accurate_counter / labelArray.size) * 100 
        print("############ Accuracy =", "{:.2f}".format(accuracy), "% ############")

if __name__ == "__main__":
    cwd = os.getcwd()
    dataLocation = cwd + '\data\parkinsons.csv'
    superVisedLearner(dataLocation)
    
    