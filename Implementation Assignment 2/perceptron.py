import numpy as np


class Perceptron:

    def __init__(self, parameters1, result1, parameters2, result2):
        self.x1 = parameters1
        self.y1 = result1
        self.x2 = parameters2
        self.y2 = result2

    def onlinePerceptron(self, maxIter=1):
        weight = np.zeros((1, self.x1.T.shape[0]))

        for i in range(0, maxIter):
            for t in range(0, self.y1.shape[0]):

                u = np.sign(np.dot(weight, self.x1[t].T))

                if (self.y1[t]*u <= 0):
                    weight = weight + self.y1[t]*self.x1[t]

            val1 = self.compAcc(self.x1, self.y1, weight)
            val2 = self.compAcc(self.x2, self.y2, weight)
            print(val1, val2)

        return weight

    def avgPerceptron(self, maxIter=1):
        weight = np.zeros((1, self.x1.T.shape[0]))
        avgWeight = np.zeros((1, self.x1.T.shape[0]))
        count, countSum = 0.0, 0.0

        for i in range(0, maxIter):

            for t in range(0, self.y1.shape[0]):

                u = np.sign(np.dot(weight, self.x1[t].T))

                if (self.y1[t]*u <= 0):
                    if (count + countSum) > 0:
                        avgWeight = self.compAvgWgt(
                            avgWeight, weight, count, countSum)
                    countSum += count
                    weight = weight + self.y1[t]*self.x1[t]
                    count = 0
                else:
                    count += 1

            val1 = self.compAcc(self.x1, self.y1, avgWeight)
            val2 = self.compAcc(self.x2, self.y2, avgWeight)
            print(val1, val2)

        if count > 0:
            avgWeight = self.compAvgWgt(
                avgWeight, weight, count, countSum)

        return avgWeight

    def correctNum(self, arr1, arr2):
        ct = 0
        for i in range(0, len(arr1)):
            if arr2[i]*np.sign(arr1[i]) > 0:
                ct += 1
        return ct

    def compAcc(self, x, y, wgt):
        val = np.dot(x, wgt.T)
        return self.correctNum(val, y)/y.shape[0]

    def compAvgWgt(self, avgWeight, weight, count, countSum):
        a = countSum / (count + countSum)
        b = count / (count + countSum)
        return a*avgWeight + b*weight
