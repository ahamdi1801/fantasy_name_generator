import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator


def show_graph(history, name):
    plt.plot(history.history['loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["train", "validation"], loc='upper left')
    p = os.path.realpath(f'./Figures/{name}')
    plt.savefig(p)
    plt.show()
