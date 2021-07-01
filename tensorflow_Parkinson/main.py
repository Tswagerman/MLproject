import pandas as pd
import numpy as np
import os

from machineLearning import model 

#Global tuples
parkinson_training_features = ([])
parkinson_training_labels = ([])
parkinson_testing_features = ([])
parkinson_testing_labels = ([])

def cleanData(parkinson_dataset):
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

def takeSampleRows(data, perc):
    return data.head(int(len(data)*(perc)))

def createDatasets(dataLocation):
    global parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels
    parkinson_dataset = pd.read_csv(dataLocation, names=["name","MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)",
                                                        "MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3",
                                                        "Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","status","RPDE","DFA","spread1",
                                                        "spread2","D2","PPE"], skiprows=[0]) #Skipping first row in csv file
    parkinson_data_clean = cleanData(parkinson_dataset)
    print("parkinson_data_clean = ", parkinson_data_clean)
    #The aim was to divide the training and testing data 80/20.
    #In order to divide the test and train sets in between subjects,
    #the percentage had to be quite precise. This resulted in an arbitrary 0.785 percent of the data.
    parkinson_training_dataset = takeSampleRows(parkinson_data_clean, 0.785) 
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

if __name__ == "__main__":
    cwd = os.getcwd()
    dataLocation = cwd + '\data\parkinson.csv'
    createDatasets(dataLocation)
    model(parkinson_training_features, parkinson_training_labels, parkinson_testing_features, parkinson_testing_labels)
    
