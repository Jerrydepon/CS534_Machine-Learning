import decisionTree as dt
import numpy as np
import random
from node import Node


class randomForest:

    def __init__(self, treeNum, ftrNum, depth, dataNum):
        self.treeNum = treeNum
        self.ftrNum = ftrNum
        self.maxDepth = depth
        self.forest = []
        self.dtClass = dt.decesionTree(depth=self.maxDepth, dataNum=dataNum)

    # Data With replacement
    # Feature Without replacement
    def bootStrap(self, df):
        rdData = df.sample(n=df.shape[0], replace=True)
        tmp = np.random.choice(range(1, df.shape[1]), self.ftrNum, replace=False)
        ftrArr = np.insert(tmp, 0, 0)
        return rdData[ftrArr]

    def buildRandomForest(self, df):
        for i in range(0, self.treeNum):
            newSubData = self.bootStrap(df)
            root = cur = Node((None, None, self.dtClass.getLabelFromLargeData(newSubData)))
            cl, cr = self.dtClass.getResultInfo(newSubData)
            self.dtClass.dicisionTree(newSubData, cur, 0, cl, cr)
            self.forest.append(root)

    def predicDataResult(self, df):
        accNum = 0
        for idx in range(0, df.shape[0]):
            voteYes = 0
            for treeRoot in self.forest:
                self.dtClass.predictRec(df.iloc[idx:idx + 1, :], treeRoot)
                voteYes += self.dtClass.getAccNum()
                self.dtClass.resetAccNum()
            # more than half tree get accurate answer
            if voteYes >= (self.treeNum / 2):
                accNum += 1
        print(accNum / df.shape[0])
