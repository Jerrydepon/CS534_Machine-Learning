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
        self.result_label = self.result * -1 + 4    # label '3' as 1, '5' as -1
        self.featureNum = data.T.shape[0] - 1       # number of features (785)
        self.exampleNum = data.shape[0]             # number of examples
        self.w = []                                 # weights
        self.wa = []                                # average weights

        # validation data
        self.feature_val = data_valid.T[1:]         # features
        self.result_val = data_valid[0]             # outcome set
        self.result_label_val = self.result_val * -1 + 4
        self.exampleNum_val = data_valid.shape[0]   # number of examples

    def validate(self, w):
        i = 0
        err_count = 0
        for i in range(self.exampleNum_val):
            x = self.feature_val[i]
            wx = np.dot(w, x)

            if (self.result_label_val[i] * wx <= 0):
                err_count += 1
        return err_count

    def printResult(self, numIter, errorCount, weight):
        print("Iter: ", numIter, end = " ")
        print("--- Prediction error: ", errorCount, end = " ")
        print("--- Acuracy: ", (self.exampleNum - errorCount) /
                                    self.exampleNum * 100)

        # make prediction on the validation samples
        err_val = self.validate(weight);

        print("Validation error: ", err_val, end = " ")
        print("--- Acuracy: ", (self.exampleNum_val - err_val) /
                                    self.exampleNum_val * 100)

    def onlinePerceptron(self, iter_limit):
        self.w = np.zeros(self.featureNum)

        for iter in range(0, iter_limit):
            err_count = 0
            for i in range(0, self.exampleNum):
                x = self.feature[i]
                wx = np.dot(self.w, x)

                if (self.result_label[i] * wx <= 0):
                    self.w += self.result_label[i] * x
                    err_count += 1

            # Output result
            self.printResult(iter+1, err_count, self.w);

        return self.w

    def averagePerceptron(self, iter_limit):
        self.w = np.zeros(self.featureNum)
        self.wa = np.zeros(self.featureNum)
        c = 0   # keeps running average weight
        sum = 0   # keeps sum of cs

        for iter in range(0, iter_limit):
            err_count = 0
            for i in range(0, self.exampleNum):
                x = self.feature[i]
                wx = np.dot(self.w, x)

                if (self.result_label[i] * wx <= 0):
                    if (sum+c > 0):
                        self.wa = (sum*self.wa + c*self.w) / (sum+c)
                    sum += c
                    self.w += self.result_label[i] * x
                    c = 0
                else :
                    c += 1

                if (self.result_label[i] * np.dot(self.wa, x) <= 0):
                    err_count += 1

            # Output result
            self.printResult(iter+1, err_count, self.wa);

        if c > 0:
            self.wa = (sum*self.wa + c*self.w) / (sum+c)

        return self.wa

    def kernelMap(self, kMap, row, col, p, x, y, x2):
        if kMap[row, col] != 0:
            return kMap[row, col]

        kMap[row, col] = (1 +
                np.dot(x[row], x2[col].T)) ** p
        return kMap[row, col]

    def sumMap(self, alpha, i, kMap, p, x, y, x2):
        sum = 0
        for j, val in alpha.items():
            sum += self.kernelMap(kMap, j, i, p, x, y, x2) * val * y[j]
        return sum

    def kernelPerceptron(self, iter_limit):
        alphaDic = {}
        power_index = 1
        kMap = np.zeros([self.exampleNum, self.exampleNum])
        kMap_v = np.zeros([self.exampleNum, self.exampleNum])

        for iter in range(0, iter_limit):
            for i in range(0, self.exampleNum):
                sum_K_alpha_y = self.sumMap(alphaDic, i, kMap, power_index,
                                    self.feature, self.result_label,
                                    self.feature, self.result_label)
                                    # sigma_j(K[j,i]alpha[j]y[j])
                u = np.sign(sum_K_alpha_y)
                if (self.result_label[i] * u <= 0):
                    alphaDic.setdefault(i, 0)
                    alphaDic[i] += 1

            err_t = self.kernelError(alphaDic, kMap, power_index,
                            self.feature, self.result_label,
                            self.exampleNum, self.feature,
                            self.result_label)
            # err_v = self.kernelError(alphaDic, kMap_v, power_index,
            #                 self.feature, self.result_label,
            #                 self.exampleNum_val, self.feature_val,
            #                 self.result_label_val)
            # print("iter", i+1, ": ", err_t, err_v)
            print("iter", iter+1, ": ", err_t)

        return alphaDic

    def kernelError(self, alpha, kMap, p, x, y, exa_size, x2, y2):
        err_count = 0
        for i in range(0, exa_size):
            sum_K_alpha_y = self.sumMap(alpha, i, kMap, p, x, y, x2)
            u = np.sign(sum_K_alpha_y)
            if (y2[i] * u <= 0):
                err_count += 1

        return (exa_size-err_count) / exa_size * 100

# =======================================================
#                          Main
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

print("\n ------------ Perceptron ------------")

pct = Perceptron(trainData, valData)
# print("\n Online Perceptron")
# q1_w = pct.onlinePerceptron(iter_limit)
# print("w of online:")
# print(q1_w)
# print("\n Average Perceptron")
# q2_w = pct.averagePerceptron(iter_limit)
# print("w of average:")
# print(q2_w)
print("\n Kernel Perceptron")
q3 = pct.kernelPerceptron(iter_limit)
print("alpha:")
print(q3)
