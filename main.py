import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from dataset import plot_loss_accuracy, output_submission, load_train_dataset, load_test_dataset, load_train_labels
from model import model, tf_model
from keras.callbacks import ModelCheckpoint, EarlyStopping


# global variables
IMG_SIZE = 60
EPOCHS = 5
BATCH_SIZE = 64


df_train = pd.read_csv('data/labels.csv')
df_test = pd.read_csv('data/sample_submission.csv')

targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse=True)
one_hot_labels = np.asarray(one_hot)

x_train = load_train_dataset(df_train, IMG_SIZE)
x_test = load_test_dataset(df_test, IMG_SIZE)
y_train = load_train_labels(df_train, one_hot_labels)

x_train_raw = np.array(x_train, np.float32) / 255.
x_test = np.array(x_test, np.float32) / 255.
y_train_raw = np.array(y_train, np.uint8)

NUM_CLASS = y_train_raw.shape[1]

print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)  # randomly flip images

# datagen.fit(X_train)

# model = model(IMG_SIZE, NUM_CLASS)
tf_model = tf_model(IMG_SIZE, NUM_CLASS)

# load pre-trained weights
tf_model.load_weights('models/vgg16_1.h5')
print('model loaded')

# Save the model according to the conditions
checkpoint = ModelCheckpoint("models/vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
#                               steps_per_epoch=X_train.shape[0] // BATCH_SIZE, epochs=EPOCHS,
#                               validation_data=(X_valid, Y_valid), workers=4, verbose=2)

tf_history = tf_model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                              steps_per_epoch=X_train.shape[0] // BATCH_SIZE, epochs=EPOCHS,
                              validation_data=(X_valid, Y_valid), workers=4, verbose=2, callbacks=[checkpoint, early])

tf_model.save_weights('models/model.h5')
print('model saved')

preds = tf_model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)

output_submission(preds, one_hot, df_test)

# plot_loss_accuracy(history)
plot_loss_accuracy(tf_history)
