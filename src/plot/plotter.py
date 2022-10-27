import matplotlib.pyplot as plt

def show_graph(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["train", "validation"], loc='upper left')
    plt.show()

    encode("Data DIY")