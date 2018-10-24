import numpy as np
import helper as hp
import datetime as dt
import perceptron as pct
import kernelPerceptron as kpt
import fileHelper as fh

maxIter = 15                     # Loop number
fileName1 = "pa2_train.csv"      # Training File name
fileName2 = "pa2_valid.csv"      # Validate File name
fileName3 = "pa2_test_no_label.csv"      # Validate File name
powNum = 2                       # Kernel function power number

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

print("\n ------------ Perceptron ------------")
pt = pct.Perceptron(par1, rst1, par2, rst2)
w = pt.onlinePerceptron(15)

print("\n ------------ AvgPerceptron ------------")
pt = pct.Perceptron(par1, rst1, par2, rst2)
w = pt.avgPerceptron(15)

print("\n ------------ kerPerceptron ------------")
print(dt.datetime.now())
kp = kpt.KernelPerceptron(par1, rst1, par2, rst2)
kp.kernelPerceptron(15,1)
print("\n ------------ kerPerceptron ------------")
kp.kernelPerceptron(15,2)
print("\n ------------ kerPerceptron ------------")
kp.kernelPerceptron(15,3)
print("\n ------------ kerPerceptron ------------")
kp.kernelPerceptron(15,7)
print("\n ------------ kerPerceptron ------------")
kp.kernelPerceptron(15,15)
print(dt.datetime.now())


par3 = hp.importCsv(fileName3, False)
result = hp.predictValue(par3, w)
fOut = fh.fileHelper("oplabel.csv")
fOut.outputResult(result)
fOut.fileClose()
