import numpy as np
import helper as hp
from datetime import date
from numpy import linalg as la
import time
from time import gmtime, strftime
import pandas as pd

class Perceptron:
    def __init__(self, data, data_valid):
        # training data
        self.feature = data.T[1:]                   # features
        self.result = data[0]                       # outcome set
        self.featureNum = data.T.shape[0] - 1       # number of features (785)
        self.exampleNum = data.shape[0]             # number of examples
        self.w = []                                 # weights
        self.wa = []                                # average weights

        # validation data
        self.feature_val = data_valid.T[1:]         # features
        self.result_val = data_valid[0]             # outcome set
        self.exampleNum_val = data_valid.shape[0]   # number of examples

    def validate(self, w):
        i = 0
        err_count = 0
        for i in range(self.exampleNum_val):
            x = self.feature_val[i]
            wx = np.dot(w, x)

            result_val_label = self.result_val[i] * -1 + 4 # label '3' as 1, '5' as -1
            if (result_val_label * wx <= 0):
                err_count += 1
        return err_count

    def onlinePerceptron(self, iter_limit):
        self.w = np.zeros(self.featureNum)

        result_label = self.result * -1 + 4 # label '3' as 1, '5' as -1

        for iter in range(0, iter_limit):
            err_count = 0
            for i in range(0, self.exampleNum):
                x = self.feature[i]
                wx = np.dot(self.w, x)

                if (result_label[i] * wx <= 0):
                    self.w += result_label[i] * x
                    err_count += 1

            # Output result
            print("Iter: ", iter+1, end = " ")
            print("--- Prediction error: ", err_count, end = " ")
            print("--- Acuracy: ", (self.exampleNum - err_count) / self.exampleNum * 100)

            # make prediction on the validation samples
            err_val = self.validate(self.w);

            print("Validation error: ", err_val, end = " ")
            print("--- Acuracy: ", (self.exampleNum_val - err_val) / self.exampleNum_val * 100)

        return self.w

    def averagePerceptron(self, iter_limit):
        self.w = np.zeros(self.featureNum)
        self.wa = np.zeros(self.featureNum)
        c = 0   # keeps running average weight
        sum = 0   # keeps sum of cs

        result_label = self.result * -1 + 4 # label '3' as 1, '5' as -1

        for iter in range(0, iter_limit):
            err_count = 0
            for i in range(0, self.exampleNum):
                x = self.feature[i]
                wx = np.dot(self.w, x)

                if (result_label[i] * wx <= 0):
                    if (sum+c > 0):
                        self.wa = (sum*self.wa + c*self.w) / (sum+c)
                    sum += c
                    self.w += result_label[i] * x
                    c = 0
                    err_count += 1
                else :
                    c += 1

            print("Iter: ", iter+1, end = " ")
            print("--- Prediction error: ", err_count, end = " ")
            print("--- Acuracy: ", (self.exampleNum - err_count) / self.exampleNum * 100)

            # make prediction on the validation samples
            err_val = self.validate(self.w);

            print("Validation error: ", err_val, end = " ")
            print("--- Acuracy: ", (self.exampleNum_val - err_val) / self.exampleNum_val * 100)

        if c > 0:
            self.wa = (sum*self.wa + c*self.w) / (sum+c)

        return self.wa

    def kernelPerceptron(init):

# =======================================================
iter_limit = 15 # limitated number of iteration

print("\n ------------ ImportDaTa ------------")
trainingFile = 'pa2_train.csv'
valFile = 'pa2_valid.csv'

trainData = pd.read_csv(trainingFile, header=None)
valData = pd.read_csv(valFile, header=None)

# add dummy column
dummy_col_t = np.ones(trainData.shape[0])
dummy_col_v = np.ones(valData.shape[0])

trainData.insert(loc=785, column='785', value=dummy_col_t)
valData.insert(loc=785, column='785', value=dummy_col_v)

# print(trainData.T[1:][1])

print("\n ------------ Perceptron ------------")

pct = Perceptron(trainData, valData)
# q1_w = pct.onlinePerceptron(iter_limit)
# print("w of online:")
# print(q1_w)
# q2_w = pct.averagePerceptron(iter_limit)
# print("w of average:")
# print(q2_w)
q3 = pct.kernelPerceptron()
