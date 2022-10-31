import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def show_graph(history, name):
    plt.plot(history.history['loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["train", "validation"], loc='upper left')
    plt.savefig(f'./Figures/{name}')
    plt.show()
