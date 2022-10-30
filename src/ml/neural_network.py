import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
import time
import numpy as np


def build_model(shape, settings):
    max_sequence_length = shape[1]
    num_chars = shape[2]

    model = Sequential()
    model.add(LSTM(settings.latent_dim,
                   input_shape=(max_sequence_length, num_chars),
                   recurrent_dropout=settings.dropout_rate))
    model.add(Dense(units=num_chars, activation='softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer)

    model.summary()

    return model


def train_model(model, X, Y, settings):

    if settings.load_model:
        model.load_weights(settings.model_path)
    else:
        start = time.time()
        print('Start training for {} epochs'.format(settings.epochs))
        history = model.fit(X, Y, epochs=settings.epochs,
                            batch_size=settings.batch_size, verbose=settings.verbosity)
        end = time.time()
        print('Finished training - time elapsed:', (end - start)/60, 'min')

    if settings.store_model:
        print('Storing model at:', settings.model_path)
        model.save(settings.model_path)


def generate_names(model, sequence, char2idx, idx2char, settings):
    new_names = []

    while len(new_names) < settings.gen_amount:
        x = np.zeros((1, len(sequence), len(idx2char)))

        for i, c in enumerate(sequence):
            x[0, i, char2idx[c]] = 1

        probs = model.predict(x, verbose=0)[0]
        probs /= probs.sum()
        next_idx = np.random.choice(len(probs), p=probs)   
        next_char = idx2char[next_idx]   
        sequence = sequence[1:] + next_char

        if next_char == '\n':
            gen_name = [name for name in sequence.split('\n')][1]

            # Discard all names that are too short
            if len(gen_name) > 2:
                # Only allow new and unique names
                if gen_name not in new_names:
                    new_names.append(gen_name.capitalize())

    return new_names

if __name__ == "__main__":
    pass
