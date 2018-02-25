from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPooling2D, Conv2D
from keras import optimizers
from keras import regularizers


def model(img_size, num_class):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))

    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
