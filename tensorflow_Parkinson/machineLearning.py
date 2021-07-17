import numpy as np
from numpy import mean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class model:
    def __init__(self, parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels):  
        self.predictionArray = []
        self.epochs = 500 #500
        self.LR = 0.4 #0.15 Learning rate
        self.DR = 0.9 #Decay rate
        self.modelRuns = 15 #3
        self.batchSize = 55 #130
        predictArray = self.__modelRuns(self.modelRuns, parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels)
        #The average testing data is used to validate the accuracy of the model.
        self.__calc_accuracy(predictArray, parkinson_testing_labels, "Testing Data ")

    def __modelRuns(self, noModels, parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels):
        #Averaging over a number of models. Committee machine
        for modelnr in range(noModels):
            #The model is created based on the training dataset.
            model = self.__createModel(parkinson_training_features, parkinson_training_labels)
            new_predictionArray = self.__make_predictions(model, parkinson_testing_features, parkinson_testing_labels)
            #print("new_predictionArray = ", new_predictionArray)
            self.predictionArray.append(new_predictionArray)
        tupleArray = tuple(map(mean, zip(*self.predictionArray)))
        predictArray = np.asarray(tupleArray) 
        #print("predictArray = ", predictArray)
        return predictArray

    def __createModel(self, parkinson_training_features, parkinson_training_labels):
        features_shape = parkinson_training_features.shape[1:]
        #Create a normalization layer and set its internal state using the training data
        #Normalization: holds the mean and standard deviation of the features
        normalizer = preprocessing.Normalization()
        normalizer.adapt(parkinson_training_features) #exposing the preprocessing layer to training data, setting its state.

        #Create model 
        input = keras.Input(shape=features_shape) #10 features, creating (10,) Tensor
        features = normalizer(input)
        hidden_layer_1 = layers.Dense(units = 10, activation="tanh") #Weights 1 * 10 + 10 * 10
        output_tensor_1 = hidden_layer_1(features)
        hidden_layer_2 = layers.Dense(units = 10, activation="tanh") #Weights 10 * 10 + 10 * 1
        output_tensor_2 = hidden_layer_2(output_tensor_1)
        output_layer = layers.Dense(units = 1, activation="sigmoid") #Weights 10 * 1 + 1
        output_tensor_3 = output_layer(output_tensor_2)
        model = keras.Model(input, output_tensor_3)

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.LR,
            decay_steps=500,
            decay_rate=self.DR)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        #Train model
        #Binary classification, hence binary crossentropy
        model.compile(optimizer=opt, loss='binary_crossentropy') # metrics=['MeanSquaredError', 'accuracy', 'AUC']
        model.summary()
        #Epoch: one forward pass and one backward pass of all the training examples
        #Batch size: number of training examples in one epoch
        model.fit(parkinson_training_features, parkinson_training_labels, batch_size=self.batchSize, epochs=self.epochs, verbose = 0) 
        
        training_prediction = self.__make_predictions(model, parkinson_training_features, parkinson_training_labels)
        predictionArray = np.asarray(training_prediction)
        self.__calc_accuracy(predictionArray, parkinson_training_labels, "Training Data")
        return model

    def __make_predictions(self, model, parkinson_features, parkinson_labels):
        predictions = model(parkinson_features)
        return predictions
        
    def __calc_accuracy(self, predictionArray, labelArray, string):
        print("\n############   ", string, "  ############")
        #print("Prediction before rounding = ", predictionArray)
        predictionArrayRounded = np.round(predictionArray,0)
        accuracy_counter = 0
        #print("PREDICTION ARRAY OF HEALTH STATUS:\n", predictionArrayRounded)
        #print("ACTUAL HEALTH STATUS\n", labelArray)
        for iteration in range(labelArray.size):
            if (predictionArrayRounded[iteration] == labelArray[iteration]):
                accuracy_counter += 1
            else:
                print("iteration = ", iteration, " prediction: ", predictionArrayRounded[iteration], " Label: ", labelArray[iteration], "PreRounding:", predictionArray[iteration])
        accuracy = (accuracy_counter / labelArray.size) * 100 
        print("############ Accuracy =", "{:.2f}".format(accuracy), "% ###########")