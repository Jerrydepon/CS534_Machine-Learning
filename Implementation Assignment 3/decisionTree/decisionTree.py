from node import Node
import numpy as np


class decesionTree:
    def __init__(self, depth, dataNum):
        self.maxDepth = depth
        self.accNum = 0
        self.dataWeights = np.ones(dataNum)

    def setDataWeight(self, arr):
        self.dataWeights = arr

    def getAccNum(self):
        return self.accNum

    def resetAccNum(self):
        self.accNum = 0

    def getLabelFromLargeData(self, df):
        val1 = df[df[0] == df[0].max()].shape[0]
        val2 = df[df[0] == df[0].min()].shape[0]
        return df[0].max() if val1 >= val2 else df[0].min()

    def setBranchLabelAndIsConverge(self, fcbll, fcblr, fcbrl, fcbrr):
        leftLable = 1 if fcbll >= fcblr else -1
        rightLable = 1 if fcbrl >= fcbrr else -1
        isLeftCon = True if fcbll == 0 or fcblr == 0 else False
        isRightCon = True if fcbrl == 0 or fcbrr == 0 else False
        return leftLable, rightLable, isLeftCon, isRightCon

    def dicisionTree(self, dataframe, treeNode, depth, cl, cr):

        branchLeft, branchRight, fcbll, fcblr, fcbrl, fcbrr = \
            self.buildTree(dataframe, treeNode, cl, cr)

        leftLable, rightLable, isLeftCon, isRightCon = \
            self.setBranchLabelAndIsConverge(fcbll=fcbll, fcblr=fcblr, fcbrl=fcbrl, fcbrr=fcbrr)

        treeNode.left = Node((None, None, leftLable))
        treeNode.right = Node((None, None, rightLable))
        # max depth is One => One node to split data
        if self.maxDepth <= depth + 1:
            return
        if not isLeftCon:
            self.dicisionTree(branchLeft, treeNode.left, depth + 1, cl=fcbll, cr=fcblr)
        if not isRightCon:
            self.dicisionTree(branchRight, treeNode.right, depth + 1, cl=fcbrl, cr=fcbrr)

    def setTreeNode(self, node, name, threshold):
        x, y, lable = node.val
        node.val = (name, threshold, lable)

    def getResultInfo(self, df):
        cl, cr = 0, 0
        for idx, row in df.iterrows():
            if row[0] == 1:
                cl += self.dataWeights[idx]
            elif row[0] == -1:
                cr += self.dataWeights[idx]
        return cl, cr

    def splitDataAndSetTreeNode(self, trnData, treeNode, fFtrColName, fThreshold):
        self.setTreeNode(treeNode, fFtrColName, fThreshold)
        branchRight = trnData[trnData[fFtrColName] >= fThreshold]
        branchLeft = trnData[trnData[fFtrColName] < fThreshold]
        return branchLeft, branchRight

    def buildTree(self, trnData, treeNode, cl, cr):
        fFtrColName, fThreshold, maxGiniNum = 0, 0.0, 0.0

        for idx in range(1, trnData.shape[1]):
            ftrColName = list(trnData)[idx]
            threshold, giniNum = self.getMaxThresholdInfo(cl, cr, trnData.sort_values(by=[ftrColName]), ftrColName)

            if giniNum > maxGiniNum:
                fFtrColName, fThreshold, maxGiniNum = ftrColName, threshold, giniNum

        branchLeft, branchRight = \
            self.splitDataAndSetTreeNode(trnData, treeNode, fFtrColName, fThreshold)
        fcbll, fcblr = self.getResultInfo(branchLeft)
        fcbrl, fcbrr = self.getResultInfo(branchRight)

        return branchLeft, branchRight, fcbll, fcblr, fcbrl, fcbrr

    def getMaxThresholdInfo(self, cl, cr, df, ftrColName):
        cbll, cblr, cbrl, cbrr = 0, 0, cl, cr
        threshold, maxGiniNum = 0.0, 0.0
        resultVal, preIdx = df.iloc[0, 0], df.index[0]

        for idx, row in df.iloc[1:].iterrows():
            preResultVal = resultVal
            cbll, cblr, cbrl, cbrr = self.computeFirstResult(preIdx, resultVal, cbll, cblr, cbrl, cbrr)
            resultVal, preIdx = row[0], idx

            if preResultVal != resultVal:
                giniNum = self.giniValue(cl, cr, cbll, cblr, cbrl, cbrr)
                if giniNum > maxGiniNum:
                    threshold, maxGiniNum = row[ftrColName], giniNum

        return threshold, maxGiniNum

    def giniValue(self, cl, cr, cbll, cblr, cbrl, cbrr):
        vl = float((2 * cbll * cblr) / ((cbll + cblr) * (cl + cr)))
        vr = float((2 * cbrl * cbrr) / ((cbrl + cbrr) * (cl + cr)))
        return 1.0 - vl - vr

    def computeFirstResult(self, idx, resultVal, cbll, cblr, cbrl, cbrr):
        if resultVal == 1:
            cbll += self.dataWeights[idx]
            cbrl -= self.dataWeights[idx]
        else:
            cblr += self.dataWeights[idx]
            cbrr -= self.dataWeights[idx]
        return cbll, cblr, cbrl, cbrr

    def compAccNumWithWeight(self, df, label):
        rt = 0
        for idx, row in df.iterrows():
            if row[0] == label:
                rt += self.dataWeights[idx]
        return rt

    def predictRec(self, data, treeNode, depth=0):
        if data.shape[0] == 0:
            return
        ftrColName, threshold, label = treeNode.val
        if ftrColName is None:
            # self.accNum += data[data[0] == label].shape[0]
            self.accNum += self.compAccNumWithWeight(data, label)
            return

        branchLeft = data[data[ftrColName] < threshold]
        branchRight = data[data[ftrColName] >= threshold]
        self.predictRec(branchLeft, treeNode.left, depth + 1)
        self.predictRec(branchRight, treeNode.right, depth + 1)
