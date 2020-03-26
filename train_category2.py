# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
# from tensorflow.keras.models import Model

from data import polyvore_dataset, DataGenerator
from utils import Config
from model import MyModel

from time import time
# import tensorflow as tf
# from tensorflow.keras.applications.mobilenet import MobileNet
# from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt 


if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_valid, X_test, y_train, y_valid, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        k = Config['debug_size']
        train_set = (X_train[:k], y_train[:k], transforms['train'])
        valid_set = (X_valid[:k], y_valid[:k], transforms['test'])
        test_set = (X_test[:k], y_test[:k], transforms['test'])
        dataset_size = {'train': k, 'test': k}
    else:
        train_set = (X_train, y_train, transforms['train'])
        valid_set = (X_valid, y_valid, transforms['test'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator =  DataGenerator(train_set, dataset_size, params)
    valid_generator = DataGenerator(valid_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)


    # Create a TensorBoard instance with the path to the logs directory
    # tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    checkpoint = ModelCheckpoint(Config['checkpoint_path'],
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=False,
                            mode='max')

    model = MyModel(n_classes).model
    # define optimizers
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
 
    model.summary()

    # training
    hist = model.fit(train_generator,
              validation_data=valid_generator, 
              epochs=Config['num_epochs'],
              callbacks=[checkpoint])

    loss = model.evaluate(test_generator)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Test accuracy: '+str(loss[1]))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig('learning_Curve.png')
    print(loss)

    # import os
    # os.system("sudo shutdown -P now")




