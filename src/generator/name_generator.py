import sys
import os
from src.ml.encoding import encode_name_list, encode_name
from src.ml.data_class import populate_data, make_dataframe, get_train_test_data
from src.ml.neural_network import build_model, compile_model, fit_model, DataGenerator
import os
# tensorflow no GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(args=sys.argv):
    filename = None
    for i, a in enumerate(args):
        if a == "-s":
            try:
                filename = args[i+1]
            except IndexError:
                print("No filename given")

    names = []
    try:
        with open(f"./data/{filename}", "r") as file:
            names = file.readlines()
    except:
        print(f"Couldn't open file with name: {filename}")
        print("use the argument: -s <filename>")
        exit(1)

    # drop the \n characters TODO: fix this for windows
    names = [n[:-1] for n in names]
    name_frame = make_dataframe(names)
    name_frame = name_frame.rename(columns={0: 'Name'})

    # make data set
    data = populate_data(name_frame)
    data.data["encoded"] = data.data.apply(
        lambda row: encode_name(row["Name"]), axis=1)


    # make train and test set generators
    (train, test) = get_train_test_data(data)
    training_generator = DataGenerator(
        train, "encoded", "encoded", batch_size=2, shuffle=True, name="train")
    validation_generator = DataGenerator(
        test, "encoded", "encoded", batch_size=2, shuffle=True, name="test")
    # print(test.info())
    # exit(0)

    model = build_model(data.data.encoded[0])
    tensorboard = compile_model(model)
    history = fit_model(model, training_generator, validation_generator, tensorboard)


if __name__ == "__main__":
    pass
