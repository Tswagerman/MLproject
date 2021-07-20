import numpy as np
from numpy import mean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class model:
    def __init__(self, parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels):  
        self.predictionArray = []
        #Tuning parameters
        self.epochs = 500 #500 Epoch: one forward pass and one backward pass of all the training examples
        self.LR = 0.45 #0.45 Learning rate
        self.DS = 50 #50 Decay steps
        self.DR = 0.98 #0.98 Decay rate
        self.modelRuns = 15 #15 Number of models being used to form the committee machine
        self.batchSize = 55 #55 Batch size: number of training examples in one epoch      
        self.__modelRuns(parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels)
       
    def __modelRuns(self, parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels):
        #Averaging over a number of models. Committee machine
        for modelnr in range(self.modelRuns):
            #The model is created based on the training dataset.
            model = self.__createAndTrainModel(parkinson_training_features, parkinson_training_labels)
            #Make predictions of the testing data using the model
            new_predictionArray = model(parkinson_testing_features)
            self.predictionArray.append(new_predictionArray)
            predictArray = np.asarray(new_predictionArray) 
            self.__calcAccuracy(predictArray, parkinson_testing_labels, "Mid-run testing")
        tupleArray = tuple(map(mean, zip(*self.predictionArray)))
        predictArray = np.asarray(tupleArray) 
        #The average testing data is used to validate the accuracy of the model.
        self.__calcAccuracy(predictArray, parkinson_testing_labels, "Testing Data ")

    def __createAndTrainModel(self, parkinson_training_features, parkinson_training_labels):
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

        #Train model
        #ExponentialDecay = initial_learning_rate * decay_rate ^ (step / decay_steps)
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.LR,
            decay_steps=self.DS,
            decay_rate=self.DR)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        #Binary classification, hence binary crossentropy
        model.compile(optimizer=opt, loss='binary_crossentropy') # metrics=['MeanSquaredError', 'accuracy', 'AUC']
        model.summary()
        model.fit(parkinson_training_features, parkinson_training_labels, batch_size=self.batchSize, epochs=self.epochs, verbose = 0) 
        
        #See how well the training data is being predicted
        training_prediction = model(parkinson_training_features)
        predictionArray = np.asarray(training_prediction)
        self.__calcAccuracy(predictionArray, parkinson_training_labels, "Training Data")
        return model
        
    def __calcAccuracy(self, predictionArray, labelArray, string):
        print("\n############   ", string, "  ############")
        predictionArrayRounded = np.round(predictionArray, 0)
        accuracy_counter = 0
        for iteration in range(labelArray.size):
            if (predictionArrayRounded[iteration] == labelArray[iteration]): #Prediction correct
                accuracy_counter += 1
            else: #Prediction incorrect
                pass
                #print("iteration = ", iteration, " prediction: ", predictionArrayRounded[iteration], " Label: ", labelArray[iteration], "PreRounding:", predictionArray[iteration])
        accuracy = (accuracy_counter / labelArray.size) * 100 
        print("############ Accuracy =", "{:.2f}".format(accuracy), "% ###########")