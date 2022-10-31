import sys
import logging
import random
import os
import numpy as np
from src.ml.encoding import build_coder_decoder, gen_valid_chars, make_training_set, encode_training_set
from src.ml.settings import gen_settings
from src.ml.neural_network import build_model, train_model, generate_names
from src.plot.plotter import show_graph

# tensorflow no GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Logging
logging.basicConfig()
logging.root.setLevel(logging.DEBUG)

def main(args=sys.argv):
    # Load settings
    settings = gen_settings(args)

    # Loading data
    filename = None
    for i, a in enumerate(args):
        if a == "-s":
            try:
                filename = args[i+1]
            except IndexError:
                print("No filename given")

    try:
        with open(f"./data/{filename}", "r") as file:
            raw_data = file.readlines()
    except:
        print(f"Couldn't open file with name: {filename}")
        print("use the argument: -s <filename>")
        exit(1)

    # coder and decoder for going from character to indices
    vc = gen_valid_chars()
    char2idx, idx2char = build_coder_decoder(vc)

    
    data = []
    for i, d in enumerate(raw_data):
        good = True
        for char in d:
            if char not in vc:
               good = False 

        if good:
            data.append(d)

    random.shuffle(data) 

    sequences, next_chars = make_training_set(data, settings)
    X, Y = encode_training_set((sequences, next_chars))
    logging.debug(X.shape)
    logging.debug(Y.shape)

    model = build_model(X.shape, settings)
    history = train_model(model, X, Y, settings)
    if history and settings.plot_loss_function:
        if not os.path.exists('Figures'):
            os.makedirs('Figures')
        show_graph(history, settings.model_path.split('/')[-1].split('.')[0])

    sequence = sequences[-1][1:] + '\n'
    generated = generate_names(model, sequence, char2idx, idx2char, settings)
    logging.debug(generated)

    # Save names
    if not os.path.exists('Output'):
        os.makedirs('Output')
    
    p = os.path.join('./Output/' + filename.split('.')[0] + settings.model_path.split('/')[-1].split('.')[0] + '_output.txt')
    with open(p, "w+") as output_file:
        for name in generated:
            output_file.write(f"{name}\n")

if __name__ == "__main__":
    pass
