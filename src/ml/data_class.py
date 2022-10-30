from dataclasses import dataclass, field
import numpy as np
import os

@dataclass(frozen=True)
class Data():
    # (number_of_sequences, max_sequence_length, valid_characters_length)
    sequences_encoding: np.array
    # (number_of_sequences, valid_characters_length)
    nex_chars_encoding: np.array


@dataclass()
class Settings():
    step_length = 1    # The step length we take to get our samples from our corpus
    epochs = 10       # Number of times we train on our full data
    batch_size = 4    # Data samples in each training step
    latent_dim = 64    # Size of our LSTM
    dropout_rate = 0.2  # Regularization with dropout
    model_path = os.path.realpath(
        './Models/model3.h5')  # Location for the model
    load_model = False  # Enable loading model from disk
    store_model = True  # Store model to disk after training
    verbosity = 1      # Print result for each epoch
    gen_amount = 10    # How many


def gen_settings(args):
    settings = Settings()
    for i, a in enumerate(args):
        if a == "-l":
            settings.load_model = True
            settings.save_model = False

    return settings
