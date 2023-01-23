import numpy as np
import pandas as pd
import tensorflow as tf
import os
from glob import glob

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation

from keras.utils.np_utils import to_categorical  # one-hot-encoding
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

lesion_type_dict = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions ',
    'df': 'Dermatofibroma',
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'vasc': 'Vascular lesions',
}

input_shape = (75, 100, 3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

base_skin_dir = os.path.join('input/HAM10000')
image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_skin_dir, '*.jpg'))}

# load metadata into memory
df = pd.read_csv('input/HAM10000_metadata.csv')
df['path'] = df['image_id'].map(image_path_dict.get)
df['cell_type'] = df['dx'].map(lesion_type_dict.get)
df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
df.head()

# load images into memory
df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))

# view the shape of the data
df['image'].map(lambda x: x.shape).value_counts()

# train/test split
features = df.drop(columns=['cell_type_idx'], axis=1)
target = df['cell_type_idx']

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20, random_state=0)

# Normalization
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

#x_train = (x_train - x_train_mean)/x_train_std
#x_test = (x_test - x_test_mean)/x_test_std

# Perform one-hot encoding on the labels

y_train = to_categorical(y_train_o)
y_test = to_categorical(y_test_o)

print(y_train.shape)

# Randomly assign images to the training and validation sets
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.1, random_state=2)

print(y_train.shape)

# Reshape image to following dims: height = 75px, width = 100px, canal = 3)
x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Make random changes in data to better generalise it
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the degree range of 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally by a fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically by a fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(x_train)

epochs = 20
batch_size = 50
cnn_history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                epochs=epochs, validation_data=(x_validate, y_validate),
                                verbose=1, steps_per_epoch=x_train.shape[0] // batch_size,
                                callbacks=[learning_rate_reduction])

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
model.save("output/model.h5")
