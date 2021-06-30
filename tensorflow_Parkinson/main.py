import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class supervisedLearner:
    def __init__(self, dataLocation):  
        self.__createDatasets(dataLocation)

    def __createDatasets(self, dataLocation):
        parkinson_dataset = pd.read_csv(dataLocation, names=["name","MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)",
                                                                   "MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3",
                                                                   "Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","status","RPDE","DFA","spread1",
                                                                   "spread2","D2","PPE"], skiprows=[0]) #Skipping first row in csv file
        parkinson_data_clean = self.__cleanData(parkinson_dataset)
        print(parkinson_data_clean.iloc[0])
        #The aim was to divide the training and testing data 80/20.
        #In order to divide the test and train sets in between subjects,
        #the percentage had to be quite precise. This resulted in an arbitrary 0.785 percent of the data.
        parkinson_training_dataset = self.__takeSampleRows(parkinson_data_clean, 0.785) 
        parkinson_training_features = parkinson_training_dataset.copy()
        parkinson_training_labels = parkinson_training_features.pop('status') #Health status of the subject (one) - Parkinson's, (zero) - healthy
        parkinson_training_features = np.array(parkinson_training_features)
        parkinson_training_labels = np.array(parkinson_training_labels)

        #The remainder of the data is used for testing the model.
        parkinson_testing_dataset = parkinson_data_clean.iloc[max(parkinson_training_dataset.index):]
        parkinson_testing_features = parkinson_testing_dataset.copy()
        parkinson_testing_labels = parkinson_testing_features.pop('status')
        parkinson_testing_features = np.array(parkinson_testing_features)
        parkinson_testing_labels = np.array(parkinson_testing_labels)
        #The model is created based on the training dataset.
        model = self.__createModel(parkinson_training_features, parkinson_training_labels)
        #The testing data is used to validate the accuracy of the model.
        self.__make_predictions(model, parkinson_testing_features, parkinson_testing_labels, "Testing Data ")
   
    def __cleanData(self, parkinson_dataset):
        #The string variable 'name' will cause errors when converting to np.array. 
        #Also, it is not relevant in order to predict whether an individual has parkinson's disease.
        parkinson_dataset.drop("name", inplace=True, axis=1) 
        #The other variables are dropped, because they do not show a correlation with Parkinson's disease
        #Based on work done shown in reference 3. (see README)
        trash_variables = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
                            "MDVP:RAP", "MDVP:PPQ", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
                            "Shimmer:APQ3", "Shimmer:APQ5", "spread1", "spread2"]
        for iteration in range(len(trash_variables)):
            parkinson_dataset.drop(trash_variables[iteration], inplace=True, axis=1) 
        return parkinson_dataset

    def __createModel(self, parkinson_training_features, parkinson_training_labels):
        features_shape = parkinson_training_features.shape[1:]
        label_variability = 2 #Either healthy or not
        #Create a normalization layer and set its internal state using the training data
        #Normalization: holds the mean and standard deviation of the features
        normalizer = preprocessing.Normalization()
        normalizer.adapt(parkinson_training_features) #exposing the preprocessing layer to training data, setting its state.

        #Create model layers
        input = keras.Input(shape=features_shape)
        features = normalizer(input)
        output = layers.Dense(label_variability, activation="softmax")(features)
        model = keras.Model(input, output)

        #Train model
        model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        model.fit(parkinson_training_features, parkinson_training_labels, epochs=100)

        self.__make_predictions(model, parkinson_training_features, parkinson_training_labels, "Training Data")
        return model

    def __takeSampleRows(self, data, perc):
        return data.head(int(len(data)*(perc)))

    def __make_predictions(self, model, parkinson_features, parkinson_labels, string):
        print("\n############   ", string, "  ############")
        predictions = model(parkinson_features)
        tf.nn.softmax(predictions)
        predictionArray = tf.argmax(predictions, axis=1)
        self.__calc_accuracy(np.array(predictionArray), parkinson_labels)
        
    def __calc_accuracy(self, predictionArray, labelArray):
        accuracy_counter = 0
        #print("PREDICTION ARRAY OF HEALTH STATUS:\n", predictionArray)
        #print("ACTUAL HEALTH STATUS\n", labelArray)
        for iteration in range(labelArray.size):
            if (predictionArray[iteration] == labelArray[iteration]):
                accuracy_counter += 1
        accuracy = (accuracy_counter / labelArray.size) * 100 
        print("############ Accuracy =", "{:.2f}".format(accuracy), "% ############")

if __name__ == "__main__":
    cwd = os.getcwd()
    dataLocation = cwd + '\data\parkinson.csv'
    supervisedLearner(dataLocation)
    
    