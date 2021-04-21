import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import os
import matplotlib.pyplot as plt
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k, models
from sklearn import metrics

# hyper parameters for model
from tensorflow.python.ops.confusion_matrix import confusion_matrix

nb_classes = 260  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 256, 256  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 5  # number of iteration the algorithm gets trained.
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation


def create_model():
    base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

    # Top Model Block
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    # add your top layer block to your base model
    return Model(base_model.input, predictions)


def train(train_data_dir, validation_data_dir, model_path):
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    model = create_model()

    # # let's visualize layer names and layer indices to see how many layers/blocks to re-train
    # # uncomment when choosing based_model_last_block_layer
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)



    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=transformation_ratio,
                                       shear_range=transformation_ratio,
                                       zoom_range=transformation_ratio,
                                       cval=transformation_ratio,
                                       horizontal_flip=True,
                                       vertical_flip=False)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    # save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
    # save_prefix='aug',
    # save_format='jpeg')
    # use the above 3 commented lines if you want to save and look at how the data augmentations look like

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc

    top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')
    callbacks_list = [
        ModelCheckpoint(top_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=0)
    ]

    # Train Simple CNN
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.n // train_generator.batch_size,
                        epochs=nb_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n // validation_generator.batch_size,
                        callbacks=callbacks_list)

    # verbose
    print("\nStarting to Fine Tune Model\n")

    # add the best weights from the train top model
    # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
    # we re-load model weights to ensure the best epoch is selected and not the last one.
    model.load_weights(top_weights_path)

    # based_model_last_block_layer_number points to the layer in your model you want to train.
    # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
    # If you want to train the last Two blocks of an Inception model it should be 172
    # layers before this number will used the pre-trained weights, layers above and including this number
    # will be re-trained based on the new data.
    for layer in model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False
    for layer in model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc
    final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')
    callbacks_list = [
        ModelCheckpoint(final_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    ]

    # fine-tune the model
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.n // train_generator.batch_size,
                        epochs=nb_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n // validation_generator.batch_size,
                        callbacks=callbacks_list)

    # save model
    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
        json_file.write(model_json)


def predict(test_dir, model_path):
    prediction_model = create_model()
    final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')
    prediction_model.load_weights(final_weights_path)
    prediction_model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=1,
                                                        shuffle=False,
                                                        class_mode='categorical')
    y_pred = prediction_model.predict(x=test_generator)
    print(test_generator.classes)
    print(y_pred.argmax(axis=1))
    cm = metrics.confusion_matrix(test_generator.labels, y_pred.argmax(axis=1))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.show()
    [test_loss, test_accuracy] = prediction_model.evaluate(x=test_generator)


if __name__ == '__main__':
    input_dir = r'./data'

    train_dir = os.path.join(os.path.abspath(input_dir), 'train')  # Inside, each class should have it's own folder
    validation_dir = os.path.join(os.path.abspath(input_dir), 'valid')  # each class should have it's own folder
    test_dir = os.path.join(os.path.abspath(input_dir), 'test')
    model_dir = os.path.join(os.path.abspath(input_dir), 'weights')

    #train(train_dir, validation_dir, model_dir)  # train model

    predict(test_dir, model_dir)

    # release memory
    k.clear_session()
