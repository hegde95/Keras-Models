from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from data import polyvore_dataset, DataGenerator
from utils import Config

from time import time
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint



if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        train_set = (X_train[:1000], y_train[:1000], transforms['train'])
        test_set = (X_test[:1000], y_test[:1000], transforms['test'])
        dataset_size = {'train': 1000, 'test': 1000}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator =  DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)


    # Create a TensorBoard instance with the path to the logs directory
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    checkpoint = ModelCheckpoint(Config['checkpoint_path'],
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=False,
                            mode='max')
    # Use GPU
    base_model = MobileNet(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(n_classes, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
 
    for layer in base_model.layers:
        layer.trainable = False
 
    # define optimizers
    model.compile(optimizer='rmsprop', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
 
    model.summary()

    # training
    model.fit(train_generator,
              validation_data=test_generator, 
              epochs=Config['num_epochs'],
              callbacks=[checkpoint,tensorboard])






