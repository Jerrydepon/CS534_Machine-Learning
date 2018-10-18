import numpy as np


def importCsv(path):

    datas = np.genfromtxt(path, delimiter=',')
    parameters = datas[:, 1:datas.shape[1]]
    result = datas[:, 0:1]
    # Insert Dummy Value
    parameters = np.c_[np.ones(datas.shape[0]), parameters]
    return (parameters, result)


def setLabels(arr, posVal, negVal):
    arr[arr == posVal] = 1
    arr[arr == negVal] = -1
