import numpy as np
import logging


def gen_valid_chars():
    letters = "abcdefghijklmnopqrstuvwxyz"
    letters += letters.upper()
    numbers = ''.join([str(n) for n in range(10)])
    special = r" !@#$%^&*()_+=-/\',.<>?;:[]{}`~" + '"' + '\n'

    valid_characters = letters + numbers + special
    valid_characters = "".join(sorted(valid_characters))

    return valid_characters


def build_coder_decoder(valid_characters):
    char2idx = dict((c, i) for i, c in enumerate(valid_characters))
    idx2char = dict((i, c) for i, c in enumerate(valid_characters))

    return char2idx, idx2char


def make_training_set(data, settings):
    valid_characters = gen_valid_chars()

    # drop the \n characters TODO: fix this for windows
    name_string = '\n'.join(data)
    # name_string = name_string.replace('\n', '#')
    names = [n[:-1] for n in data]

    # Use longest name as sequence window
    max_sequence_length = max([len(name) for name in names])

    logging.debug(f'Total indexed characters: {len(valid_characters)}')
    logging.debug(f'Number of names: {len(names)}')
    logging.debug(f'Length of name string: {len(name_string)}')
    logging.debug(f'Length longest name: {max_sequence_length}')

    sequences = []
    next_chars = []

    for i in range(0, len(name_string) - max_sequence_length, settings.step_length):
        sequences.append(name_string[i: i + max_sequence_length])
        next_chars.append(name_string[i + max_sequence_length])

    for i in range(10):
        logging.debug(
            f'S=<{sequences[i]}>    C=<{next_chars[i]}>'.replace('\n', ' '))

    return sequences, next_chars


def encode_training_set(training_set, valid_characters=None, char2idx=None, idx2char=None):
    if not valid_characters:
        valid_characters = gen_valid_chars()
    if not char2idx or not idx2char:
        char2idx, idx2char = build_coder_decoder(valid_characters)
    # unpack training set
    sequences, next_chars = training_set    

    num_sequences = len(sequences)
    num_chars = len(valid_characters)
    # TODO: could fail
    max_sequence_length = len(sequences[0])

    X = np.zeros((num_sequences, max_sequence_length,
                 num_chars), dtype=np.bool)
    Y = np.zeros((num_sequences, num_chars), dtype=np.bool)

    for i, sequence in enumerate(sequences):
        for j, char in enumerate(sequence):
            X[i, j, char2idx[char]] = 1
        
        Y[i, char2idx[next_chars[i]]] = 1

    return X, Y

if __name__ == "__main__":
    test_name = "Bob"
    name_encoding = np.array(encode_name(test_name))
    decode_name(name_encoding)
