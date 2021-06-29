import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class superVisedLearner:
    def __init__(self, dataLocation):
        self.parkinsons_dataset = pd.read_csv(dataLocation, names=["name","MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","status","RPDE","DFA","spread1","spread2","D2","PPE"], skiprows=[0])
        self.__createDatasets()

    def __createDatasets(self):
        #The string variable will cause errors when converting to np.array.
        self.parkinsons_dataset.drop("name", inplace=True, axis=1) 
        #Since each subject is divided in 6 or 7 rows, the percentage had to be quite precise 
        #in order to divide the test and train sets in between subject. 
        #Resulting in an arbitrary 0.785 percent of the data.
        parkinsons_training_dataset = self.__takeSampleRows(self.parkinsons_dataset, 0.785) 
        parkinsons_training_features = parkinsons_training_dataset.copy()
        parkinsons_training_labels = parkinsons_training_features.pop('status') #Health status of the subject (one) - Parkinson's, (zero) - healthy
        parkinsons_training_features = np.array(parkinsons_training_features)
        parkinsons_training_labels = np.array(parkinsons_training_labels)
        self.__createModel(parkinsons_training_features, parkinsons_training_labels)
        self.__preprocessing(parkinsons_training_features, parkinsons_training_labels)
        #The remainder of the data is used for testing the model
        parkinsons_testing_dataset = self.parkinsons_dataset.iloc[max(parkinsons_training_dataset.index):]

    def __createModel(self, parkinsons_training_features, parkinsons_training_labels):
        parkinsons_model = tf.keras.Sequential([layers.Dense(64),layers.Dense(1)])
        parkinsons_model.compile(loss = tf.losses.MeanSquaredError(),
                          optimizer = tf.optimizers.Adam())
        parkinsons_model.fit(parkinsons_training_features, parkinsons_training_labels, epochs=10)

    def __takeSampleRows(self, data, perc):
        return data.head(int(len(data)*(perc)))

    def __preprocessing(self, parkinsons_training_features, parkinsons_training_labels):
        normalize = preprocessing.Normalization()
        normalize.adapt(parkinsons_training_features)
        norm_parkinsons_model = tf.keras.Sequential([
            normalize,
            layers.Dense(64),
            layers.Dense(1)])
        norm_parkinsons_model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam())
        norm_parkinsons_model.fit(parkinsons_training_features, parkinsons_training_labels, epochs=10)

if __name__ == "__main__":
    cwd = os.getcwd()
    dataLocation = cwd + '\data\parkinsons.csv'
    superVisedLearner(dataLocation)
    
    