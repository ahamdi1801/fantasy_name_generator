import numpy as np


def gen_valid_chars():
    letters = "abcdefghijklmnopqrstuvwxyz"
    letters += letters.upper()
    numbers = ''.join([str(n) for n in range(10)])
    special = r" !@#$%^&*()_+=-/\',.<>?;:[]{}`~" + '"'

    valid_characters = letters + numbers + special
    valid_characters = "".join(sorted(valid_characters))

    return valid_characters


def encode_name_list(list_of_names):
    output = []
    for name in list_of_names:
        encoded_name = encode_name(name)
        output.append(encode_name(name))
    output = [o for o in output if o is not None]

    return output


def encode_name(name, valid_characters=None, max_len=36):
    if not valid_characters:
        valid_characters = gen_valid_chars()
    if len(name) > max_len:
        return None
    vll = len(valid_characters)
    output = []
    for i, c in enumerate(name):
        ic = valid_characters.index(c)
        cc = np.zeros(vll)
        cc[ic] = 1
        output.append(cc)

    if i < max_len:
        output += [np.zeros(vll) for i in range(max_len - len(name))]

    return np.array(output)


def decode_name(input_matrix, valid_characters=None):
    if not valid_characters:
        valid_characters = gen_valid_chars()
    # vll = len(valid_characters)
    name = ''
    # Todo: could optimise this
    for v in input_matrix:
        z = list(zip(sorted(valid_characters), v))
        fz = list(filter(lambda x: x[1] == 1, z))
        if fz == []:
            break
        letter = fz[0][0]
        name += letter

    return name


if __name__ == "__main__":
    test_name = "Bob"
    name_encoding = np.array(encode_name(test_name))
    decode_name(name_encoding)
