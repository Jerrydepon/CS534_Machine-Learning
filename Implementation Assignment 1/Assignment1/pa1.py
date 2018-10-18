import numpy as np
import helper as hp
from datetime import date
from numpy import linalg as la
import time
from time import gmtime, strftime


# Linear Regression class
class LinearRegression:

    # Intial global values
    def __init__(self, datas, result, validateDatas, validateResult, fileName):
        self.x = datas                              # training examples
        self.y = result                             # outcome price
        self.dataNum = len(datas)                   # number of data
        self.thetas = []                            # array of parameters
        self.vx = validateDatas                     # Validate Data
        self.vy = validateResult                    # Validate Result
        self.fileName = fileName + \
            strftime("%Y%m%d%H%M%S", gmtime())      # Out files name
        self.file1 = None                           # File for training data
        self.file2 = None                           # File for validate data

    # Compute the value by using matrix only
    # it is more quickly than using for loop
    def wgtVals(self):
        hVal = np.dot(self.x, self.thetas)
        return np.dot(self.x.T, (hVal - self.y))

    def OutPutSEEResult(self, isValidate, normVal):
        sseVal1 = (hp.sse(self.y, hp.predictVals(self.thetas, self.x)))
        self.file1.write(str(sseVal1)+","+str(normVal))
        self.file1.write("\n")

        if isValidate:
            sseVal2 = (hp.sse(self.vy, hp.predictVals(self.thetas, self.vx)))
            self.file2.write(str(sseVal2))
            self.file2.write("\n")
        return sseVal1

    # Using csv files to plot data
    def fileOpen(self, isValidate):
        self.file1 = open(self.fileName+"-train.csv", "a+")
        if isValidate:
            self.file2 = open(self.fileName+"-validate.csv", "a+")

    # Using csv files to plot data
    def fileClose(self, isValidate):
        self.file1.close()
        if isValidate:
            self.file2.close()

    # learning rate                 => alpha=10**-1
    # Converge condition            => limit=0.5
    # Iteration number              => maxIter=1000
    # Regulization value            => lam=0.0
    # is Out put Validate sse file  => isValidate=False
    def gradientDescent(self, alpha=10**-1, limit=0.5, maxIter=1000, lam=0.0, isValidate=False):
        converged, count = False, 1

        # Initail point of weights values
        self.thetas = np.zeros((self.x.T.shape[0], 1))
        self.fileOpen(isValidate)
        while not converged:

            # Compute Weight Value
            gdVal = self.wgtVals()
            # Compute Regulization Values exclude bias values
            regVal = lam*self.thetas
            regVal[0][0] = 0
            gdVal = gdVal + regVal
            # Compute norm value of weight value
            normVal = la.norm(gdVal)

            sseVal = self.OutPutSEEResult(isValidate, normVal)

            if normVal < limit:
                break

            # Compute Regulization Values exclude bias values
            self.thetas = self.thetas - (alpha * (gdVal))

            count += 1
            # SSEVal is explode than or Count value eqaul to max iteration value
            # than return values
            if count == maxIter or sseVal == float('Inf') or sseVal == float('NaN'):
                break

        print("Iteration : " + str(count))
        print("Final SSE Value : " +
              str(hp.sse(self.y, hp.predictVals(self.thetas, self.x))))
        self.fileClose(isValidate)
        return self.thetas


# =============================================================================
# ################ Main Function ################
# =============================================================================
alphaVal = 10 ** (-5)               # learning rate
limit = 0.5                         # convergence condition
maxIter = 300000                     # limitation of iteration
lam = 0.0                           # regularization coefficient
outPutFile = "pa1_result_"          # Out put file name
isValidate = True                   # is Out put validation result
isNormalize = True                  # is Normalize input date
trainingFile = "PA1_train.csv"      # Training file name
ValidateFile = "PA1_dev.csv"        # Validate file name

print("Learning Rate:"+str(alphaVal))
print("Convergence condition(norm): "+str(limit))
print("Limitation of iteration: "+str(maxIter))
print("Regularization coefficient: "+str(lam))


print("\n ------------ ImportDaTa ------------")
dataSet = hp.importCsv(trainingFile, isNormalize)
testSet = hp.importCsv(ValidateFile, isNormalize)


print("\n ------------ LinearRegression ------------")
x1, y1 = np.matrix(dataSet[0]), np.matrix(dataSet[1]).T
x2, y2 = np.matrix(testSet[0]), np.matrix(testSet[1]).T
lg = LinearRegression(x1, y1, x2, y2, outPutFile+str(alphaVal)+"--")
w = lg.gradientDescent(alphaVal, float(limit), maxIter, float(lam), isValidate)
print("Weight Value:")
print(w)
 