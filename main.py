import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import model
from dataset import plot_loss_accuracy, output_submission, load_train_dataset, load_test_dataset, load_train_labels


# global variables
IMG_SIZE = 48
EPOCHS = 1
BATCH_SIZE = 64


df_train = pd.read_csv('data/labels.csv')
df_test = pd.read_csv('data/sample_submission.csv')

targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse=True)
one_hot_labels = np.asarray(one_hot)

x_train = load_train_dataset(df_train, IMG_SIZE)
y_train = load_train_labels(df_train, one_hot_labels)
x_test = load_test_dataset(df_test, IMG_SIZE)

x_train_raw = np.array(x_train, np.float32) / 255.
x_test = np.array(x_test, np.float32) / 255.
y_train_raw = np.array(y_train, np.uint8)

NUM_CLASS = y_train_raw.shape[1]

print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)

model = model(IMG_SIZE, NUM_CLASS)

# load pre-trained weights
# model.load_weights('models/model.h5')
# print('model loaded')

history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=EPOCHS, batch_size=BATCH_SIZE,
                    verbose=1)

model.save_weights('models/model.h5')
print('model saved')

preds = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)

output_submission(preds, one_hot, df_test)

plot_loss_accuracy(history)
