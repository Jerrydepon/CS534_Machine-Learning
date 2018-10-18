import numpy as np
import helper as hp
from datetime import date
from numpy import linalg as la
import time
from time import gmtime, strftime
import pandas

class Perceptron:
    def __init__(self, data, data_valid):
        # training data
        self.data = data                        # training data
        self.result = data[0]                   # outcome set
        self.featureNum = data.T.shape[0] - 1   # number of features (785)
        self.exampleNum = data.shape[0]         # number of examples
        self.w = []                             # weights

        # validation data
        self.data_val = data_valid              # validation data
        self.result_val = data_valid[0]         # outcome set
        self.exampleNum_val = data_valid.shape[0] # number of examples

    def validate(self, w):
        i = 0
        err_count = 0
        for i in range(self.exampleNum_val):
            x = self.data_val.T[i][1:]
            wx = np.dot(self.w, x)

            result_val_label = self.result_val[i] * -1 + 4 # label '3' as 1, '5' as -1
            if (result_val_label * wx <= 0):
                err_count += 1
        return err_count

    def onlinePerceptron(self, iter_limit):
        self.w = np.zeros(self.featureNum)
        iter = 0
        # while iter < iter_limit:
        for iter in range(iter_limit):
            i = 0
            err_count = 0
            for i in range(self.exampleNum):
                x = self.data.T[i][1:]
                wx = np.dot(self.w, x)

                result_label = self.result[i] * -1 + 4 # label '3' as 1, '5' as -1
                if (result_label * wx <= 0):
                    self.w += result_label * x
                    # print("w: ", self.w)
                    err_count += 1
                i += 1

            print("Iter: ", iter+1, end = " ")
            print("--- Prediction error: ", err_count, end = " ")
            print("--- Acuracy: ", (self.exampleNum - err_count) / self.exampleNum * 100)

            # make prediction on the validation samples
            err_val = self.validate(self.w);

            print("Validation error: ", err_val, end = " ")
            print("--- Acuracy: ", (self.exampleNum_val - err_val) / self.exampleNum_val * 100)


# =======================================================
iter_limit = 15

print("\n ------------ ImportDaTa ------------")
trainingFile = 'pa2_train.csv'
valFile = 'pa2_valid.csv'
trainData = pandas.read_csv(trainingFile, header=None)
valData = pandas.read_csv(valFile, header=None)

print("\n ------------ Perceptron ------------")

pct = Perceptron(trainData, valData)
w = pct.onlinePerceptron(iter_limit)
