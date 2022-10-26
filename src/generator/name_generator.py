import sys
import os
from src.ml.encoding import encode_name_list



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

    encoded_names = encode_name_list(names)

