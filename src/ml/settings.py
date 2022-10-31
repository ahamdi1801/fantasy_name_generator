from dataclasses import dataclass
import numpy as np
import os

# DEFAULT SETTINGS
# step_length = 1    
# epochs = 20       
# batch_size = 32    
# latent_dim = 64    
# dropout_rate = 0.2  
# model_path = os.path.realpath(
#     './Models/model.h5')  
# load_model = False  
# store_model = True  
# verbosity = 1      
# gen_amount = 10    
# plot_loss_function = False

@dataclass()
class Settings():
    step_length = 1    # The step length we take to get our samples from our corpus
    epochs = 5       # Number of times we train on our full data
    batch_size = 32    # Data samples in each training step
    latent_dim = 64    # Size of our LSTM
    dropout_rate = 0.2  # Regularization with dropout
    model_path = os.path.realpath(
        './Models/model_50_fn.h5')  # Location for the model
    load_model = False  # Enable loading model from disk
    store_model = True  # Store model to disk after training
    verbosity = 1      # Print result for each epoch
    gen_amount = 10    # How many
    plot_loss_function = True # Plots the loss function


def gen_settings(args):
    settings = Settings()
    for i, a in enumerate(args):
        if a == "-l":
            settings.load_model = True
            settings.save_model = False

    return settings
