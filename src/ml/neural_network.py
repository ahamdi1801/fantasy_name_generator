import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import TensorBoard
from time import time
import numpy as np


def build_model(t):
    model = Sequential()
    tr = t.shape[1]
    tc = t.shape[0]
    model.add(Dense(tr*tc, input_shape=(tc, tr), activation="relu"))
    model.add(Flatten())
    model.add(Dense(int(tr*tc*0.2), activation="relu"))
    model.add(Dense(int(tr*tc*0.1), activation="relu"))
    model.add(Dense(int(tr*tc*0.05), activation="relu"))
    model.add(Dense(94, activation="linear"))

    return model


def compile_model(m):
    m.compile(loss='mse', optimizer='adam')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    return tensorboard


def fit_model(m, training_generator, validation_generator, tensorboard, epochs=10):
    # TODO: deprecated
    history = m.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[tensorboard]
    )

    return history


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, input_col, label_col, batch_size=2, shuffle=True, name=""):
        self.data = data
        self.batch_size = batch_size
        self.input_col = input_col
        self.label_col = label_col
        self.indecies = [i for i in range(len(data))]
        self.shuffle = shuffle
        self.batch_no = 0
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, c_index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_rows = self.indecies[c_index *
                                   self.batch_size:(c_index+1)*self.batch_size]

        # Generate data
        inputs, labels = self.__data_generation(batch_rows)

        return inputs, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indecies)

    def __data_generation(self, rows):
        input_lst = []
        label_lst = []
        for i in rows:
            input_lst.append(self.data.iloc[i][self.input_col])
            label_lst.append(self.data.iloc[i][self.label_col])
        return np.asarray(input_lst), np.asarray(label_lst)


if __name__ == "__main__":
    pass
