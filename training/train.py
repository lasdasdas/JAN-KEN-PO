import os
import random

#  Use pip
import cv2
import glob
import numpy as np

# Needs also tensorflow
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


def load_images_from_folder(folder, data, labelname, outshape):
    """
    Imports a folder of images into the data vector
    with a given category
    """
    init = len(data)
    for filename in glob.glob(folder+"/*.jpg"):
        imgtmp = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(imgtmp, (outshape[0], outshape[1]),
                         interpolation=cv2.INTER_LANCZOS4)
        if img is not None:
            data.append((img, labelname, filename))
    print("Finished importing from " + folder +
          " with "+str(len(data)-init)+" new images")


def format(data, img_shape, labels):
    """
    Format the data vector data to the Keras
    structure, image and label wise.
    """
    npdata = np.zeros((len(data), img_shape[1], img_shape[0], img_shape[2]),
                      dtype=np.float32)
    nplabels = np.zeros((len(data), labels), dtype=np.float32)
    for i in range(len(data)):
        npdata[i] = data[i][0]
        nplabels[i, data[i][1]] = 1.0
    return npdata, nplabels


data = []
# import the data
num_classes = 3
batch_size = 38
epochs = 5000
mult = 3
img_shape = [64*mult, 48*mult, 3]  # 640×480
labelnames = ["r", "p", "s"]

load_images_from_folder("./2_rock", data, 0, img_shape)
load_images_from_folder("./cam/2_rock", data, 0, img_shape)
load_images_from_folder("./cam2/rock", data, 0, img_shape)
load_images_from_folder("./cam3/rock", data, 0, img_shape)

# Separation of data into tran and test
random.shuffle(data)
x_train, y_train = format(data[700:], img_shape, num_classes)
x_test, y_test = format(data[0:700], img_shape, num_classes)


# data is no longer needed, lets clue the GC to delete it and free some memory
del data

model = keras.applications.vgg19.VGG19(include_top=False,
                                       weights='imagenet',
                                       input_shape=(img_shape[1], img_shape[0],
                                                    img_shape[2]), classes=3)

# Add the classifier layer into the loaded model
for layer in model.layers[:16]:
    layer.trainable = False

x = model.output
x = Dropout(0.4)(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.6)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.6)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)
model_final = Model(input=model.input, output=predictions)
print(model_final.summary())

checkpoint = ModelCheckpoint("saved_weight.h5", monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', min_delta=0,
                      patience=45, verbose=1, mode='auto')

model_final.compile(loss="categorical_crossentropy",
                    optimizer=keras.optimizers.RMSprop(lr=0.0001,
                                                       rho=0.9,
                                                       epsilon=None,
                                                       decay=1e-6),
                    metrics=["accuracy"])


datagen = ImageDataGenerator(
        rescale=1/255.0,
        samplewise_center=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        channel_shift_range=0.14,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
        )


model_final.fit_generator(datagen.flow(x_train, y_train,
                                       batch_size=batch_size),
                          steps_per_epoch=len(x_train)/batch_size,
                          epochs=epochs,
                          validation_data=datagen.flow(x_test, y_test),
                          callbacks=[checkpoint, early])
