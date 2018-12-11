import pandas as pd


def importCsv(path, isSplit=True):
    df = pd.read_csv(path, header=None)
    df[[0]] = (df[[0]] - 4) * -1
    return df
