import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class model:
    def __init__(self, parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels):  
        self.predictionArray = []
        self.noModels = 7
        modelnr = 1
        for modelnr in range(self.noModels):
            #The model is created based on the training dataset.
            model = self.__createModel(parkinson_training_features, parkinson_training_labels)
            new_predictionArray = self.__make_predictions(model, parkinson_testing_features, parkinson_testing_labels, "Testing Data ")
            i = 0
            for i in range(len(new_predictionArray)):
                if i == 0:
                    self.predictionArray = new_predictionArray.copy()
                else:
                    self.predictionArray[i] = (self.predictionArray[i] * (modelnr - 1) + new_predictionArray[i])/modelnr
        #The average testing data is used to validate the accuracy of the model.
        self.__calc_accuracy(np.array(self.predictionArray), parkinson_testing_labels)
        print(parkinson_testing_labels)

    def __createModel(self, parkinson_training_features, parkinson_training_labels):
        features_shape = parkinson_training_features.shape[1:]
        label_variability = 2 #Either healthy or not
        num_of_attributes = 11
        #Create a normalization layer and set its internal state using the training data
        #Normalization: holds the mean and standard deviation of the features
        normalizer = preprocessing.Normalization()
        normalizer.adapt(parkinson_training_features) #exposing the preprocessing layer to training data, setting its state.

        #Create model 
        input = keras.Input(shape=features_shape) #11 inputs
        #print("shape = ", features_shape)
        #features = normalizer(input)
        #print(features)
        hidden_layer_1 = layers.Dense(11, input_dim=(10,), activation="sigmoid") #Weights 11*11 + 11*11
        output_tensor_1 = hidden_layer_1(input)
        weight_1 = hidden_layer_1.get_weights()
        #print("Weight1 = ", weight_1)
        #print("Weight1 = ", len(weight_1[1]))
        hidden_layer_2 = layers.Dense(11, input_dim=(10,), activation="sigmoid") #Weights 11*11 + 11*1
        output_tensor_2 = hidden_layer_2(output_tensor_1)
        weight_2 = hidden_layer_2.get_weights()
        #print("Weight2 = ", weight_2)
        #print("Weight2 = ", len(weight_2[1]))
        output_layer = layers.Dense(1,input_dim=(1,), activation="sigmoid")
        output_tensor_3 = output_layer(output_tensor_2)
        weight_3 = output_layer.get_weights()
        #print("Weight3 = ", weight_3)
        model = keras.Model(input, output_tensor_3)

        #Train model
        #model.compile(loss = keras.losses.mean_squared_error,
         #     optimizer = keras.optimizers.Adadelta(),
          #    metrics=['accuracy'])
        #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        #    initial_learning_rate=1e-2,
        #    decay_steps=10000,
        #    decay_rate=0.9)
        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer='adam', loss='binary_crossentropy') # metrics=['MeanSquaredError', 'accuracy', 'AUC']
        #model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        model.summary()
        print(parkinson_training_labels)
        model.fit(parkinson_training_features, parkinson_training_labels, epochs=500)
        self.__make_predictions(model, parkinson_training_features, parkinson_training_labels, "Training Data")
        return model

    def __make_predictions(self, model, parkinson_features, parkinson_labels, string):
        print("\n############   ", string, "  ############")
        predictions = model(parkinson_features)
        print(predictions)
        print("Label = ", parkinson_labels)
        #tf.nn.softmax(predictions)
        #predictionArray = tf.argmax(predictions, axis=1)
        predictionArray = np.array(predictions)
        print(predictionArray)
        #print(predictionArray)
        return predictionArray
        
    def __calc_accuracy(self, predictionArray, labelArray):
        predictionArray = np.round(predictionArray,0)
        accuracy_counter = 0
        #print("PREDICTION ARRAY OF HEALTH STATUS:\n", predictionArray)
        #print("ACTUAL HEALTH STATUS\n", labelArray)
        for iteration in range(labelArray.size):
            if (predictionArray[iteration] == labelArray[iteration]):
                accuracy_counter += 1
            else:
                print("############")
                print("iteration = ", iteration)
                print("predictionArray = ", predictionArray[iteration])
                print("labelArray = ", labelArray[iteration])
                print("############")
        accuracy = (accuracy_counter / labelArray.size) * 100 
        print("############ Accuracy =", "{:.2f}".format(accuracy), "% ############")