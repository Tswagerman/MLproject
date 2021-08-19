import numpy as np
import os
from numpy import mean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.models import load_model

class model:
    def __init__(self, parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels):  
        self.prediction_array = []
        self.number_models_saved = 0
        self.__nrModelsSaved()
        #Stored some of the best performing models,
        #the average accuracy is about 85% over the testing data. These models differ in tuning parameters
        #Boolean to use '.hdf5' models that are stored in 'cwd\models'
        self.use_saved_models = self.__yesOrNo()
        #self.use_saved_models = False
        #Tuning parameters
        #ExponentialDecay = initial_learning_rate * decay_rate ^ (step / decay_steps)
        self.epochs = 500     #500  | Epoch: one forward pass and one backward pass of all the training examples
        self.LR = 0.1        #0.45 | Learning rate
        self.DS = 10         #50   | Decay steps
        self.DR = 0.9        #0.98 | Decay rate
        self.model_runs = 11  #15   | Number of models being used to form the committee machine
        self.batch_size = 55 #55   | Batch size: number of training examples in one epoch  
        self.__modelRuns(parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels)
       
    def __modelRuns(self, parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels):
        #Averaging over a number of models. Committee machine
        for modelnr in range(self.model_runs): 
            if (self.use_saved_models == True):
                #Pre-stored models
                model = self.__loadModel(modelnr)
            else:
                #The model is created based on the training dataset.
                model = self.__createAndTrainModel(parkinson_training_features, parkinson_training_labels)
            #Make predictions of the testing data using the model
            new_prediction_array = model(parkinson_testing_features)
            self.prediction_array.append(new_prediction_array)
            predictArray = np.asarray(new_prediction_array) 
            self.__calcAccuracy(predictArray, parkinson_testing_labels, "Mid-run testing")
        tuple_array = tuple(map(mean, zip(*self.prediction_array)))
        predict_array = np.asarray(tuple_array) 
        print(predict_array)
        #The average testing data is used to validate the accuracy of the model.
        self.__calcAccuracy(predict_array, parkinson_testing_labels, "Testing Data ")

    def __createAndTrainModel(self, parkinson_training_features, parkinson_training_labels):
        features_shape = parkinson_training_features.shape[1:]
        #Create a normalization layer and set its internal state using the training data
        #Normalization: holds the mean and standard deviation of the features
        normalizer = preprocessing.Normalization()
        normalizer.adapt(parkinson_training_features) #exposing the preprocessing layer to training data, setting its state.

        #Create model 
        input = keras.Input(shape=features_shape) #10 features, creating (10,) Tensor
        features = normalizer(input)
        hidden_layer_1 = layers.Dense(units = 5, activation="tanh") #Weights 1 * 5 + 5 * 5
        output_tensor_1 = hidden_layer_1(features)
        hidden_layer_2 = layers.Dense(units = 5, activation="tanh") #Weights 5 * 5 + 5 * 1
        output_tensor_2 = hidden_layer_2(output_tensor_1)
        output_layer = layers.Dense(units = 1, activation="sigmoid") #Weights 5 * 1 + 1
        output_tensor_3 = output_layer(output_tensor_2)
        model = keras.Model(input, output_tensor_3)

        #Train model
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.LR,
            decay_steps=self.DS,
            decay_rate=self.DR)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        #Binary classification, hence binary crossentropy
        model.compile(optimizer=opt, loss='binary_crossentropy') # metrics=['MeanSquaredError', 'accuracy', 'AUC']
        model.summary()
        model.fit(parkinson_training_features, parkinson_training_labels, batch_size=self.batch_size, epochs=self.epochs, verbose = 0) 
        
        #See how well the training data is being predicted
        training_prediction = model(parkinson_training_features)
        prediction_array = np.asarray(training_prediction)
        self.__calcAccuracy(prediction_array, parkinson_training_labels, "Training Data")
        return model

    def __nrModelsSaved(self):
        cwd = os.getcwd()
        saveLocation = cwd + '\model'
        list = os.listdir(saveLocation) 
        self.number_models_saved = len(list)
        
    def __calcAccuracy(self, prediction_array, label_array, string):
        print("\n############   ", string, "  ############")
        prediction_array_rounded = np.round(prediction_array, 0).flatten()
        accuracy_counter = 0
        #print('pr = ', prediction_arrayRounded)
        #print('label = ', label_array)
        for iteration in range(label_array.size):
            if (prediction_array_rounded[iteration] == label_array[iteration]): #Prediction correct
                accuracy_counter += 1
            else: #Prediction incorrect
                #pass
                print("iteration = ", iteration, " prediction: ", prediction_array_rounded[iteration], " Label: ", label_array[iteration], "PreRounding:", prediction_array[iteration])
        accuracy = (accuracy_counter / label_array.size) * 100 
        print("############ Accuracy =", "{:.2f}".format(accuracy), "% ###########")

    def __saveModel(self, model):
        filename = "model\m%s.hdf5" % self.number_models_saved
        model.save(filename)
        self.__nrModelsSaved()

    def __loadModel(self, number):
        filename = "model\m%s.hdf5" % number
        model = load_model(filename)
        return model

    def __yesOrNo(self):
        while True:
            reply = str(input('Do you want to use saved models? (y/n): ')).lower().strip()
            if reply[:1] == 'y':
                return True
            if reply[:1] == 'n':
                return False