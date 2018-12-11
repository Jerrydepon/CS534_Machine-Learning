import helper as hp
import decisionTree as dt
import randomForest as ft
import adaboost as ada
from node import Node
import datetime
import warnings
import numpy as np

# =============================================================================
# ################ Main Function ################
# =============================================================================
isDecisionTree = True                       # Is Running decision tree
dtDepthArr = [1]                             # Decision tree depth array => [ (depth num) ]
isAdaboost = True                           # Is Running adaboost
adaboostArr = [1]                            # adaboost running number => [ (ada num) ]
isRandomForest = True                       # Is Running random forest
rdForestArr = [(9, 10, 1)]                   # adaboost running number => [ (depth=n, ftrNum=m, treeNum=d) ]
fileName1 = "pa3_train_reduced.csv"          # Training File name
fileName2 = "pa3_valid_reduced.csv"          # Validate File name
warnings.filterwarnings("error")
print("\n ------------ ImportDaTa ------------")
trainData = hp.importCsv(fileName1)
validateData = hp.importCsv(fileName2)

if isDecisionTree:
    for i in dtDepthArr:
        print("\n ------------ Build DT ------------{0}".format(datetime.datetime.now()))
        dtClass = dt.decesionTree(i, trainData.shape[0])
        root1 = root2 = cur = Node((None, None, dtClass.getLabelFromLargeData(trainData)))
        cl, cr = dtClass.getResultInfo(trainData)
        dtClass.dicisionTree(trainData, cur, 0, cl, cr)

        print("\n ------------ Acc - TrainingData ------------{0}".format(datetime.datetime.now()))
        dtClass.setDataWeight(np.ones(trainData.shape[0]))
        dtClass.predictRec(trainData, root1)
        print(dtClass.getAccNum() / trainData.shape[0])
        dtClass.resetAccNum()

        print("\n ------------ Acc - ValidateDaTa ------------{0}".format(datetime.datetime.now()))
        dtClass.setDataWeight(np.ones(validateData.shape[0]))
        dtClass.predictRec(validateData, root2)
        print(dtClass.getAccNum() / validateData.shape[0])
        dtClass.resetAccNum()

if isRandomForest:
    for d, m, n in rdForestArr:
        print("\n ------------ Build Forest{0} ------------{1}".format(n, datetime.datetime.now()))
        ftClass = ft.randomForest(treeNum=n, ftrNum=m, depth=d, dataNum=trainData.shape[0])
        ftClass.buildRandomForest(df=trainData)
        ftClass.predicDataResult(df=trainData)
        ftClass.predicDataResult(df=validateData)

if isAdaboost:
    for l in adaboostArr:
        print("\n ------------ Adaboost-{0} ------------{1}".format(l, datetime.datetime.now()))
        adaClass = ada.adaboost(depth=9, lNum=l, dataNum=trainData.shape[0])
        adaClass.runAdaboost(df=trainData)
        print(adaClass.computeFinalAccNumRate(df=trainData))
        print(adaClass.computeFinalAccNumRate(df=validateData))
