import numpy as np


def importCsv(path, isSplit=True):

    datas = np.genfromtxt(path, delimiter=',')
    if isSplit:
        return splitDataSet(datas)
    else:
        return np.c_[np.ones(datas.shape[0]), datas]


def splitDataSet(datas):
    parameters = datas[:, 1:datas.shape[1]]
    result = datas[:, 0:1]
    # Insert Dummy Value to first column
    parameters = np.c_[np.ones(datas.shape[0]), parameters]
    return (parameters, result)


def setLabels(arr, posVal, negVal):
    arr[arr == posVal] = 1
    arr[arr == negVal] = -1


def predictValue(x, wgt): 
    return np.sign(np.dot(x, wgt.T))
