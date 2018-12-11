import numpy as np


class KernelPerceptron(object):

    def __init__(self, parameters1, result1, parameters2, result2):
        """
        Args:
            x1,y1 (matrix): training data
            x2,y2 (matrix): validate data
        """
        self.x1 = parameters1
        self.y1 = result1
        self.x2 = parameters2
        self.y2 = result2

    def kernelPerceptron(self, maxIter=1, powNum=2):
        """
        Args:
            maxIter (int): number of counting converge 
            powNum (str): EX 2 = quadratic space. 
        Returns:
            dic : numbers of mistake value
        """
        numData = self.x1.shape[0]
        alphaDic = {}
        kMap1 = np.zeros([numData, numData])
        kMap2 = np.zeros([numData, numData])

        for x in range(0, maxIter):
            tt = 0
            for i in range(0, numData):

                u = self.compSignValue(
                    alphaDic, self.x1, self.y1, self.x1, kMap1, i, powNum)
                if (self.y1[i]*u <= 0):
                    alphaDic.setdefault(i, 0.0)
                    alphaDic[i] += 1.0
                    tt += 1

            val = self.compKerAcc(
                alphaDic, self.x1, self.y1, self.x2, self.y2, kMap2, powNum)
            print((1-(tt/self.y1.shape[0])), val)

        return alphaDic

    def compSignValue(self, alphaDic, xw, yw, xs, kMap, i, powNum):
        """
        Args:
            alphaDic (dict): key => which data is err, val => how many time 
            xw (str): for computing weight value.
            yw (str): for computing weight value.
            xs (str): maybe is mistake data.
            kMap (str): kernel map store the data.
            i (str): represent computed mistake data
            powNum (str): EX 2 = quadratic space. 

        Returns:
            int : which class. +1 , 0 , -1.  
        """
        sumNum = 0.0
        for j, val in alphaDic.items():
            sumNum += val*yw[j] * self.setKMapValue(kMap, j, i, xw, xs, powNum)
        return np.sign(sumNum)

    def setKMapValue(self, kMap, row, col, xw, xs, p):
        # Return value which have computed before
        if kMap[row, col] != 0:
            return kMap[row, col]

        kMap[row, col] = (np.dot(xw[row], xs[col].T) + 1.0) ** p
        return kMap[row, col]

    def compKerAcc(self, alphaDic, xw, yw, xs, ys, kMap, powNum):
        err = 0
        numData = xs.shape[0]
        for i in range(0, numData):
            u = self.compSignValue(alphaDic, xw, yw, xs, kMap, i, powNum)
            if (ys[i]*u <= 0):
                err += 1

        return (1-(err/numData))

    # def compKplableResult(self, alphaDic, xs, powNum):
    #     numData = xs.shape[0]
    #     xw, yw = self.x1, self.y1
    #     kMap = np.zeros([self.x1.shape[0], self.x1.shape[0]])
    #     ys = []
    #     for i in range(0, numData):
    #         ys.append(self.compSignValue(
    #             alphaDic, xw, yw, xs, kMap, i, powNum)[0,0])
    #     return ys
