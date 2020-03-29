#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 04:47:28 2020

@author: shashank
"""

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten,Input,GlobalAveragePooling2D, concatenate
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model
from utils import Config

class MyModel:
    def __init__(self):
        if Config['Custom_model']:
            self.model = self.get_custom_model()
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