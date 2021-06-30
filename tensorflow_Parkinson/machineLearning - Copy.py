import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class model:
    def __init__(self, parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels):  
        #The model is created based on the training dataset.
        model = self.__createModel(parkinson_training_features, parkinson_training_labels)
        #The testing data is used to validate the accuracy of the model.
        self.__make_predictions(model, parkinson_testing_features, parkinson_testing_labels, "Testing Data ")

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