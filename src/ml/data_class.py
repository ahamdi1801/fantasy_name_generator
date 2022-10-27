from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Data():
    data: pd.DataFrame
    train_pct: float = .9


def populate_data(data):
    training_data = Data(data)
    return training_data


def make_dataframe(l):
    return pd.DataFrame(l)


def get_train_test_data(data):
    train = data.data.sample(frac=data.train_pct)
    test = data.data[[d not in train.index for d in data.data.index]]
    return (train, test)
