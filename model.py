import tensorflow.keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import *
from keras.optimizers import *

class MyModel:
    def __init__(self,n_classes):
        self.model = self.get_model(n_classes)
        return None
    
    def get_model(self,n):
        Input_1 = Input(shape=(224, 224, 3), name='Input_1')
        Convolution2D_1 = Conv2D(4, kernel_size=3, padding='same', activation='relu')(Input_1)
        Convolution2D_2 = Conv2D(4, kernel_size=3, padding= 'same' ,activation= 'relu')(Convolution2D_1)
        MaxPooling2D_1 = MaxPooling2D()(Convolution2D_2)
        
        Convolution2D_5 = Conv2D(8, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_1)
        Convolution2D_6 = Conv2D(8, kernel_size=3, padding='same', activation='relu')(Convolution2D_5)
        MaxPooling2D_2 = MaxPooling2D()(Convolution2D_6)
        
        Convolution2D_7 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_2)
        Convolution2D_8 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_7)
        Convolution2D_11 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_8)
        MaxPooling2D_3 = MaxPooling2D()(Convolution2D_11)
        
        Convolution2D_9 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_3)
        Convolution2D_10 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(Convolution2D_9)
        Convolution2D_12 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_10)
        MaxPooling2D_4 = MaxPooling2D(name='MaxPooling2D_4')(Convolution2D_12)
        
        Convolution2D_13 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_4)
        Convolution2D_14 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(Convolution2D_13)
        Convolution2D_16 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_14)
        MaxPooling2D_5 = MaxPooling2D(name='MaxPooling2D_5')(Convolution2D_16)
        
        Flatten_1 = Flatten()(MaxPooling2D_5)
        Dense_1 = Dense(512,activation= 'relu' )(Flatten_1)
        Dropout_1 = Dropout(0.2)(Dense_1)
        Dense_2 = Dense(512,activation= 'relu' )(Dropout_1)
        Dropout_2 = Dropout(0.2)(Dense_2)
        Dense_3 = Dense(n,activation= 'softmax' )(Dropout_2)
        
        model = Model([Input_1],[Dense_3])
        return model
    
    
    
    def get_optimizer(self):
    	return Adam()
    
    def is_custom_loss_function(self):
    	return False
    
    def get_loss_function(self):
    	return 'categorical_crossentropy'
    
    def get_batch_size(self):
    	return 32
    
    def get_num_epoch(self):
    	return 10
    
    def get_data_config(self):
    	return '{"kfold": 1, "samples": {"validation": 1323, "training": 5292, "split": 1, "test": 0}, "datasetLoadOption": "batch", "shuffle": true, "numPorts": 1, "mapping": {"Label": {"port": "OutputPort0", "type": "Categorical", "options": {}, "shape": ""}, "Filename": {"port": "InputPort0", "type": "Image", "options": {"Normalization": false, "Scaling": 1, "rotation_range": 0, "Resize": false, "width_shift_range": 0, "horizontal_flip": false, "Height": 28, "height_shift_range": 0, "Width": 28, "shear_range": 0, "vertical_flip": false, "Augmentation": true, "pretrained": "None"}, "shape": ""}}, "dataset": {"type": "public", "samples": 6615, "name": "Soda Bottles"}}'
