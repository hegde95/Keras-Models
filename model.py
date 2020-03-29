from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten,Input,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model, Sequential
from utils import Config

class MyModel:
    def __init__(self,n_classes):
        if Config['Custom_model']:
            self.model = self.get_custom_model(n_classes)
        else:
            self.model = self.get_transfer_model(n_classes)
        return None
    
    def get_custom_model(self,n):
        Input_1 = Input(shape=(224, 224, 3), name='Input_1')
        Convolution2D_1 = Conv2D(4, kernel_size=3, padding='same', activation='relu')(Input_1)
        Convolution2D_2 = Conv2D(4, kernel_size=3, padding= 'same' ,activation= 'relu')(Convolution2D_1)
        Convolution2D_2 = BatchNormalization()(Convolution2D_2)
        MaxPooling2D_1 = MaxPooling2D()(Convolution2D_2)
        
        Convolution2D_5 = Conv2D(8, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_1)
        Convolution2D_6 = Conv2D(8, kernel_size=3, padding='same', activation='relu')(Convolution2D_5)
        Convolution2D_6 = BatchNormalization()(Convolution2D_6)
        MaxPooling2D_2 = MaxPooling2D()(Convolution2D_6)
        
        Convolution2D_7 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_2)
        Convolution2D_8 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_7)
        Convolution2D_11 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_8)
        Convolution2D_11 = BatchNormalization()(Convolution2D_11)
        MaxPooling2D_3 = MaxPooling2D()(Convolution2D_11)
        
        Convolution2D_9 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_3)
        Convolution2D_10 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(Convolution2D_9)
        Convolution2D_12 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_10)
        Convolution2D_12 = BatchNormalization()(Convolution2D_12)
        MaxPooling2D_4 = MaxPooling2D(name='MaxPooling2D_4')(Convolution2D_12)
        
        Convolution2D_13 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_4)
        Convolution2D_14 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(Convolution2D_13)
        Convolution2D_16 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_14)
        Convolution2D_16 = BatchNormalization()(Convolution2D_16)
        MaxPooling2D_5 = MaxPooling2D(name='MaxPooling2D_5')(Convolution2D_16)
        
        Flatten_1 = Flatten()(MaxPooling2D_5)
        Dense_1 = Dense(512,activation= 'relu' )(Flatten_1)
        # Dropout_1 = Dropout(0.2)(Dense_1)
        Dense_2 = Dense(512,activation= 'relu' )(Dense_1)
        # Dropout_2 = Dropout(0.2)(Dense_2)
        Dense_3 = Dense(n,activation= 'softmax' )(Dense_2)
        
        model = Model([Input_1],[Dense_3])
        return model
    
    # def get_custom_model2(self,n):
    #     Input_1 = Input(shape=(224, 224, 3), name='Input_1')
    #     x = Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3))(Input_1)
    #     x = BatchNormalization()(x)
        
    #     x = Conv2D(8, kernel_size=(3, 3), activation='relu')(x)
    #     x = BatchNormalization()(x)
    #     x = MaxPooling2D(pool_size=(2, 2))(x)
    #     x = Dropout(0.25)(x)
        
    #     x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.25)(x)
        
    #     x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    #     x = BatchNormalization()(x)
    #     x = MaxPooling2D(pool_size=(2, 2))(x)
    #     x = Dropout(0.25)(x)
        
    #     x = Flatten()(x)
        
    #     x = Dense(512, activation='relu')(x)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.5)(x)
        
    #     x = Dense(256, activation='relu')(x)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.5)(x)
        
    #     x = Dense(n, activation='softmax')(x)
    #     # Dense_3 = Dense(n,activation= 'softmax' )(Dropout_2)
            
    #     model = Model([Input_1],[x])
    #     return model
    
    def get_transfer_model(self,n):
            base_model = MobileNet(weights='imagenet', include_top=False)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.2)(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.2)(x)   
            predictions = Dense(n, activation = 'softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
         
            for layer in base_model.layers:
                layer.trainable = False
            return model