import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import pandas as pd
import tensorflow as tf

class Model:

    def __init__(self, SEQUENCE_SIZE = 6):
        
        # Model Buiding
        self.__model = tf.keras.models.Sequential()
        self.__SEQUENCE_SIZE = SEQUENCE_SIZE

    def __soft_voting(self, predict_proba):

        probabilities = np.array([(1 - predict_proba).sum(), predict_proba.sum()])
        
        proba_real = probabilities[1] / probabilities.sum()
        proba_fake = 1 - proba_real

        return np.array([proba_fake, proba_real])
        
    def buid_model(self, BATCH_SIZE = 1):

        # Adding an input layer
        self.__model.add(tf.keras.layers.InputLayer(input_shape=(self.__SEQUENCE_SIZE, 26), batch_size=BATCH_SIZE))

        # Adding recurrent neural layer
        self.__model.add(tf.keras.layers.GRU(units=150, input_shape=(BATCH_SIZE, self.__SEQUENCE_SIZE, 26), activation='relu'))

        # Adding ANN layers
        self.__model.add(tf.keras.layers.Dense(units=50, input_shape=(BATCH_SIZE, 100)))
        self.__model.add(tf.keras.layers.BatchNormalization())
        self.__model.add(tf.keras.layers.Activation('relu'))
        self.__model.add(tf.keras.layers.Dense(units=1, input_shape=(BATCH_SIZE, 25), activation='sigmoid'))

    def load_weights(self, file_path):

        # Adding pre-trained weight to the model
        self.__model.load_weights(file_path)

    def predict(self, dataframe):

        if self.__SEQUENCE_SIZE == 1:
            
            data = dataframe.to_numpy()

        else:

            total_rows = dataframe.to_numpy()
            no_of_windows = math.ceil(len(total_rows) / self.__SEQUENCE_SIZE)

            if no_of_windows <= 1:

                data.append(total_rows)
            
            else:

                data = []
                curr_window = 0
                low = 0 ; high = self.__SEQUENCE_SIZE
                
                while curr_window < no_of_windows:

                    data.append(total_rows[low : high])

                    curr_window += 1
                    low = high
                    high += self.__SEQUENCE_SIZE

            padded_data  = tf.keras.preprocessing.sequence.pad_sequences(data, padding='post', dtype='float32')
            predict_proba = self.__model.predict(padded_data, verbose= None)
            final_proba = self.__soft_voting(predict_proba)

            if 1.00 > final_proba[1] > 0.8:
                return f'Audio is Real'
            
            elif 0.8 > final_proba[1] > 0.5:
                return f'Audio Seems Real with {final_proba[1] * 100: .2f}% Probability'
            
            elif 0.5 > final_proba[1] > 0.2:
                return f'Audio Seems Fake with {final_proba[0] * 100: .2f}% Probability'
            
            elif 0.2 > final_proba[1] > 0:
                return 'Audio is Fake'