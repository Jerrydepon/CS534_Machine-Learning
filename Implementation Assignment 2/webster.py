import numpy as np
import helper as hp

# Linear Regression class


class Perceptron:

    def __init__(self, parameters1, result1, parameters2, result2):
        self.x1 = parameters1
        self.y1 = result1
        self.x2 = parameters2
        self.y2 = result2
        self.weight = []
        self.avgWeight = []

    def sign(self, w, x):
        result = np.dot(w, x)
        return 1 if result > 0 else -1

    def ct(self, arr1, arr2):
        ct = 0
        for i in range(0, len(arr1)):
            if arr1[i] > 0 and arr2[i] > 0:
                ct += 1
            elif arr1[i] < 0 and arr2[i] < 0:
                ct += 1
        return ct

    def compAcc(self, wgt):
        val1, val2 = 0, 0
        rt1 = np.dot(self.x1, wgt.T)
        val1 = self.ct(rt1, self.y1)/self.y1.shape[0]

        if self.x2 is not None:
            rt2 = np.dot(self.x2, wgt.T)
            val2 = self.ct(rt2, self.y2)/self.y2.shape[0]

        return (val1, val2)

    def onlinePerceptron(self, maxIter=1):
        self.weight = np.zeros((1, self.x1.T.shape[0]))

        for i in range(0, maxIter):
            for t in range(0, self.y1.shape[0]):

                u = self.sign(self.weight, self.x1[t].T)

                if (self.y1[t]*u <= 0):
                    self.weight = self.weight + self.y1[t]*self.x1[t]

            (val1, val2) = self.compAcc(self.weight)
            print((val1, val2))

        return self.weight

    def avgPerceptron(self, maxIter=1):
        self.weight = np.zeros((1, self.x1.T.shape[0]))
        self.avgWeight = np.zeros((1, self.x1.T.shape[0]))
        count, countSum = 0.0, 0.0

        for i in range(0, maxIter):

            for t in range(0, self.y1.shape[0]):

                u = self.sign(self.weight, self.x1[t].T)

                if (self.y1[t]*u <= 0):
                    if (count + countSum) > 0:
                        a = countSum / (count + countSum)
                        b = count / (count + countSum)
                        self.avgWeight = a*self.avgWeight + b*self.weight
                    countSum += count
                    self.weight = self.weight + self.y1[t]*self.x1[t]
                    count = 0
                else:
                    count += 1

            (val1, val2) = self.compAcc(self.avgWeight)
            print((val1, val2))
        if count > 0:
            self.avgWeight = (countSum*self.avgWeight) + \
                (count*self.weight) / (count + countSum)

        return self.avgWeight

    def kernelPerceptron(self, maxIter=1):
        numData = self.x1.shape[0]
        numFtr = self.x1.shape[1]
        self.weight = np.zeros((1, self.x1.T.shape[0]))

        for i in range(0, maxIter):
            for t in range(0, self.y1.shape[0]):

                u = self.sign(self.weight, self.x1[t].T)

                if (self.y1[t]*u <= 0):
                    self.weight = self.weight + self.y1[t]*self.x1[t]

            (val1, val2) = self.compAcc(self.weight)
            print((val1, val2))

        return self.weight

# =============================================================================
# ################ Main Function ################
# =============================================================================
maxIter = 15
fileName1 = "pa2_train.csv"      # File name one
fileName2 = "pa2_valid.csv"      # File name one

print("\n ------------ ImportDaTa ------------")
(par1, rst1) = hp.importCsv(fileName1)
(par2, rst2) = hp.importCsv(fileName2)
hp.setLabels(rst1, 3, 5)
hp.setLabels(rst2, 3, 5)
par1 = np.matrix(par1)
rst1 = np.matrix(rst1)
par2 = np.matrix(par2)
rst2 = np.matrix(rst2)

print("\n ------------ Perceptron ------------")
pt = Perceptron(par1, rst1, par2, rst2)
w = pt.onlinePerceptron(15)

print("\n ------------ AvgPerceptron ------------")
pt = Perceptron(par1, rst1, par2, rst2)
w = pt.avgPerceptron(15)
