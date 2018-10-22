import numpy as np
import helper as hp
import datetime as dt

# Linear Regression class


class Perceptron:

    def __init__(self, parameters1, result1, parameters2, result2):
        self.x1 = parameters1
        self.y1 = result1
        self.x2 = parameters2
        self.y2 = result2
        self.weight = []
        self.avgWeight = []

    def correctNum(self, arr1, arr2):
        ct = 0
        for i in range(0, len(arr1)):
            if arr2[i]*np.sign(arr1[i]) > 0:
                ct += 1
        return ct

    def compAcc(self, wgt):
        val1, val2 = 0, 0
        rt1 = np.dot(self.x1, wgt.T)
        val1 = self.correctNum(rt1, self.y1)/self.y1.shape[0]

        if self.x2 is not None:
            rt2 = np.dot(self.x2, wgt.T)
            val2 = self.correctNum(rt2, self.y2)/self.y2.shape[0]

        return (val1, val2)

    def onlinePerceptron(self, maxIter=1):
        self.weight = np.zeros((1, self.x1.T.shape[0]))

        for i in range(0, maxIter):
            for t in range(0, self.y1.shape[0]):

                u = np.sign(np.dot(self.weight, self.x1[t].T))

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

                u = np.sign(np.dot(self.weight, self.x1[t].T))

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

    def setKMapValue(self, kMap, row, col, xw, xs, p=2):
        if kMap[row, col] != 0:
            return kMap[row, col]

        # print (np.dot(xw[row], xs[col].T))
        # print (np.sum(np.dot(xw[row].T, xs[col])))
        kMap[row, col] = (np.dot(xw[row], xs[col].T) + 1) ** p
        return kMap[row, col]

    def compSignValue(self, alphaDic, xw, yw, xs, kMap, i, powNum):
        sumNum = 0
        for j, val in alphaDic.items():
            sumNum += val*yw[j] * self.setKMapValue(kMap, j, i, xw, xs, powNum)
        return sumNum

    def kernelPerceptron(self, maxIter=1):
        powNum = 2
        numData = self.x1.shape[0]
        alphaDic = {}
        kMap1 = np.zeros([numData, numData])
        kMap2 = np.zeros([numData, numData])

        for x in range(0, maxIter):
            for i in range(0, numData):

                u = np.sign(self.compSignValue(
                    alphaDic, self.x1, self.y1, self.x1, kMap1, i, powNum))
                if (self.y1[i]*u <= 0):
                    alphaDic.setdefault(i, 0)
                    alphaDic[i] += 1

            val1 = self.compKerAcc(
                alphaDic, self.x1, self.y1, self.x1, self.y1, kMap1, powNum)
            val2 = self.compKerAcc(
                alphaDic, self.x1, self.y1, self.x2, self.y2, kMap2, powNum)
            print(val1, val2)

        return alphaDic

    def compKerAcc(self, alphaDic, xw, yw, xs, ys, kMap, powNum):
        err = 0
        numData = xs.shape[0]
        for i in range(0, numData):
            u = np.sign(self.compSignValue(
                alphaDic, xw, yw, xs, kMap, i, powNum))
            if (ys[i]*u <= 0):
                err += 1

        return (1-(err/numData))


# ===============================================================================================================================================
# ################ Main Function ################
# ===============================================================================================================================================
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
print(par1.shape)
print(par2.shape)

# print("\n ------------ Perceptron ------------")
# pt = Perceptron(par1, rst1, par2, rst2)
# w = pt.onlinePerceptron(1)

# print("\n ------------ AvgPerceptron ------------")
# pt = Perceptron(par1, rst1, par2, rst2)
# w = pt.avgPerceptron(15)


print("\n ------------ kerPerceptron ------------")
print(dt.datetime.now())
pt = Perceptron(par1, rst1, par2, rst2)
w = pt.kernelPerceptron(1)
# print(dt.datetime.now())
print(w)
