import decisionTree as dt
import numpy as np
from node import Node
import math
import datetime


class adaboost:

    def __init__(self, depth, lNum, dataNum):
        self.wArr = []
        self.L = lNum
        self.dtClass = dt.decesionTree(depth=depth, dataNum=dataNum)
        self.alphaSet = []
        self.treeSet = []
        self.weightInitValue = 1

    def runAdaboost(self, df):
        self.wArr = np.ones(df.shape[0]) * self.weightInitValue
        # self.wArr = np.ones(df.shape[0]) * self.weightInitValue / df.shape[0]
        # self.normalizeWeightArr(df.shape[0])
        val = self.dtClass.getLabelFromLargeData(df)

        for l in range(0, self.L):
            self.dtClass.setDataWeight(self.wArr)
            cl, cr = self.dtClass.getResultInfo(df)

            root = cur = Node((None, None, val))
            self.dtClass.dicisionTree(df, cur, 0, cl, cr)

            alpha, errIdxArr = self.getAlphaVal(df, root)
            self.updWeightArr(df, alpha, errIdxArr)
            self.normalizeWeightArr(df.shape[0])
            self.storeImpData(root=root, alpha=alpha)

    def updTrainData(self, df):
        df = df.mul(self.wArr, axis=0)
        df[0] = np.sign(df[0])

    # W(1) * Alpha
    # Weight Update Rule => W(2) = W(1)* exp(Alpha * h(Xi) * -Yi)
    def computeFinalAccNumRate(self, df):
        fAccNum, dataNum = 0, df.shape[0]
        for dataIdx in range(0, dataNum):
            weight, tmp = self.weightInitValue, 0
            for l in range(0, self.L):
                self.dtClass.predictRec(df.iloc[dataIdx:dataIdx + 1, :], self.treeSet[l])
                isCorrect = 1 if self.dtClass.getAccNum() != 0 else -1
                # isCorrect = -1 if dataIdx in self.errIdxSet[l] else 1
                tmp += isCorrect * self.alphaSet[l]
                self.dtClass.resetAccNum()
            if np.sign(tmp) > 0:
                fAccNum += 1
        return fAccNum / dataNum

    def normalizeWeightArr(self, dataNum):
        self.wArr = self.wArr / np.sum(self.wArr)
        self.wArr = self.wArr * dataNum

    def compNewWeight(self, alpha, isCorrect):
        return math.exp(alpha * isCorrect * -1)

    def getAlphaVal(self, df, root):
        totalNum = sum(self.wArr)
        accNum, errIdx = self.getAccAndErrInfo(df=df, root=root)
        errRate = (totalNum - accNum) / totalNum
        print("Error Rate = {0} Alpha = {1}".format(errRate, self.compAlphaByErr(errRate)))
        if errRate == 0:
            print("Loop {0} times find error rate = 0")
        return self.compAlphaByErr(errRate), errIdx

    def updWeightArr(self, df, alpha, errIdxArr):
        for dataIdx in range(0, df.shape[0]):
            isCorrect = -1 if dataIdx in errIdxArr else 1
            self.wArr[dataIdx] *= self.compNewWeight(alpha, isCorrect)

    def getAccNum(self, df, root):
        self.dtClass.resetAccNum()
        self.dtClass.predictRec(df, root)
        return self.dtClass.getAccNum()

    def compAlphaByErr(self, errRate):
        return (1 / 2) * math.log((1 - errRate) / errRate)

    def storeImpData(self, root, alpha):
        self.alphaSet.append(alpha)
        self.treeSet.append(root)

    def getAccAndErrInfo(self, df, root):
        errIdx, accNum = [], 0
        for dataIdx in range(0, df.shape[0]):
            self.dtClass.predictRec(df.iloc[dataIdx:dataIdx + 1, :], root)
            accNum += self.dtClass.getAccNum()
            if self.dtClass.getAccNum() == 0:
                errIdx.append(dataIdx)
            self.dtClass.resetAccNum()
        return accNum, errIdx
