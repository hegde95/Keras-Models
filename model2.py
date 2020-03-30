#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 04:47:28 2020

@author: shashank
"""

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten,Input,GlobalAveragePooling2D, concatenate, Lambda, BatchNormalization
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model, Sequential
from utils import Config
import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K

class MyModel:
    def __init__(self):
        if Config['Custom_model']:
            self.model = self.get_siamese_model()
        else:
            self.model = self.get_transfer_model()
        return None
        
    def get_custom_model(self):
        Input_1 = Input(shape=(224, 224, 3), name='Input_1')
        Input_2 = Input(shape=(224, 224, 3), name='Input_2')
        
        x1 = Conv2D(4, kernel_size=3, padding='same', activation='relu')(Input_1)
        x1 = Conv2D(4, kernel_size=3, padding= 'same' ,activation= 'relu')(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(8, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = Conv2D(8, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Flatten()(x1)
        x1 = Dense(20,activation= 'relu' )(x1)
        x1 = Dropout(0.2)(x1)
        model1 = Model(inputs=Input_1, outputs=x1)

        x2 = Conv2D(4, kernel_size=3, padding='same', activation='relu')(Input_2)
        x2 = Conv2D(4, kernel_size=3, padding= 'same' ,activation= 'relu')(x2)
        x2 = MaxPooling2D()(x2)
        x2 = Conv2D(8, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = Conv2D(8, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = MaxPooling2D()(x2)
        x2 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = MaxPooling2D()(x2)
        x2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = MaxPooling2D()(x2)
        x2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = MaxPooling2D()(x2)
        x2 = Flatten()(x2)
        x2 = Dense(20,activation= 'relu' )(x2)
        x2 = Dropout(0.2)(x2)
        model2 = Model(inputs=Input_2, outputs=x2)
        
        x3 = concatenate([model1.output, model2.output])
        x3 = Dense(20,activation= 'relu' )(x3)
        x3 = Dropout(0.2)(x3)
        
        predictions = Dense(1, activation = 'sigmoid')(x3)

        model = Model(inputs=[model1.input, model2.input], outputs=predictions)
        return model
    
    def get_transfer_model(self):
        Input_1 = Input(shape=(224, 224, 3), name='Input_1')
        Input_2 = Input(shape=(224, 224, 3), name='Input_2')
        
        base_model1 = MobileNet(weights='imagenet', include_top=False, input_tensor = Input_1)
        for layer in base_model1.layers:
            layer.trainable = False
            
        x1 = base_model1.output
        x1 = GlobalAveragePooling2D()(x1)
        x1 = Dense(512, activation='relu')(x1)
        x1 = Dropout(0.2)(x1)
        x1 = Dense(512, activation='relu')(x1)
        x1 = Dropout(0.2)(x1) 
        model1 = Model(inputs=Input_1, outputs=x1)
        
        for layer in model1.layers:
            layer._name = layer.name + str("_1")   
            
        base_model2 = MobileNet(weights='imagenet', include_top=False, input_tensor = Input_2)
        for layer in base_model2.layers:
            layer.trainable = False        
        
        x2 = base_model2.output
        x2 = GlobalAveragePooling2D()(x2)
        x2 = Dense(512, activation='relu')(x2)
        x2 = Dropout(0.2)(x2)
        x2 = Dense(512, activation='relu')(x2)
        x2 = Dropout(0.2)(x2)   
        model2 = Model(inputs=Input_2, outputs=x2)
        for layer in model2.layers:
            layer._name = layer.name + str("_2")  
    
        combined = concatenate([model1.output, model2.output])
        x3 = Dense(512, activation='relu')(combined)
        x3 = Dropout(0.2)(x3)
        x3 = Dense(512, activation='relu')(x3)
        x3 = Dropout(0.2)(x3)
        predictions = Dense(1, activation = 'sigmoid')(x3)

        model = Model(inputs=[model1.input, model2.input], outputs=predictions)
        return model

    def get_siamese_model(self):
        """
            Model architecture
        """
        input_shape = (224, 224, 3)
        # Define the tensors for the two input images
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        
        # Convolutional Neural Network
        # model = Sequential()
        # model.add(Conv2D(4, (3,3), activation='relu', input_shape=input_shape, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), kernel_regularizer=l2(2e-4)))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D())
        # model.add(Conv2D(8, (3,3), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), bias_initializer=RandomNormal(mean=0.5, stddev=1e-2), kernel_regularizer=l2(2e-4)))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D())
        # model.add(Conv2D(16, (3,3), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), bias_initializer=RandomNormal(mean=0.5, stddev=1e-2), kernel_regularizer=l2(2e-4)))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D())
        # model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), bias_initializer=RandomNormal(mean=0.5, stddev=1e-2), kernel_regularizer=l2(2e-4)))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D())
        # model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), bias_initializer=RandomNormal(mean=0.5, stddev=1e-2), kernel_regularizer=l2(2e-4)))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D())
        # model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), bias_initializer=RandomNormal(mean=0.5, stddev=1e-2), kernel_regularizer=l2(2e-4)))
        # model.add(Flatten())
        # model.add(Dense(512, activation='sigmoid', kernel_regularizer=l2(1e-3), kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),bias_initializer=RandomNormal(mean=0.5, stddev=1e-2)))

        model = Sequential()
        model.add(Conv2D(4, (3,3), activation='relu', input_shape=input_shape, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(8, (3,3), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), bias_initializer=RandomNormal(mean=0.5, stddev=1e-2)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(16, (3,3), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), bias_initializer=RandomNormal(mean=0.5, stddev=1e-2)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), bias_initializer=RandomNormal(mean=0.5, stddev=1e-2)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), bias_initializer=RandomNormal(mean=0.5, stddev=1e-2)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2), bias_initializer=RandomNormal(mean=0.5, stddev=1e-2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),bias_initializer=RandomNormal(mean=0.5, stddev=1e-2)))
        
        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)
        
        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        
        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = Dense(1,activation='sigmoid',bias_initializer=RandomNormal(mean=0.5, stddev=1e-2))(L1_distance)
        
        # Connect the inputs with the outputs
        siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
        
        # return the model
        return siamese_net

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)