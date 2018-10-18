from datetime import datetime
import numpy as np 
import math


"""
Get SSE given ground truth and prediction vectors
"""


def sse(realVal, predVal):
    return np.sum(np.power((realVal-predVal), 2.0)) / 2.0


"""
Takes in weights vector w, and a dataset matrix which contains input rows excluding output column
and returns prediction vector.
"""


def predictVals(thetas, datas):
    return np.dot(datas, thetas)


"""
Normalize Values between 0 to 1
Matrix -> Matrix
"""


def NormalizeData(matrix):
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    maxs - mins

    for col in range(1, len(matrix[0])):
        for row in range(0, len(matrix)):
            matrix[row][col] = (matrix[row][col] - mins[col]
                                ) / (maxs[col] - mins[col])
    return matrix


"""
Import data from CSV files
"""


def importCsv(path, isNormalize, delimiter=",", isHead=True):
    x, y = [], []

    with open(path) as f:
        lines = f.readlines()

        for line in lines:

            if isHead:
                isHead = False
                continue
            arr = line.split(delimiter)

            # Remove data - bedroom = 33, price 6.4
            if float(arr[3]) >= 30.0:
                continue

            # Modified our datetime value
            arr[2] = float(datetime.strptime(
                arr[2], '%m/%d/%Y').strftime("%Y%m%d")) 
            year = float(math.floor(arr[2]/10000))
            month = float(math.floor((arr[2]-year*10000) / 100))
            day = float(arr[2]-year*10000-month*100)  
            diff_day = (2018*365 + 5*30 + 31) - (year*365 + month*30 + day)


            # Setting our dataset
            y.append(float(arr.pop().replace("\n", "")))
            x.append([float(arr[0])     # Dummy
                      # , float(arr[1]) # ID
                      # , float(arr[2]) # date
                      , diff_day        # modified date
                      , float(arr[3])   # bedrooms
                      , float(arr[4])   # bathrooms
                      , float(arr[5])   # sqft_living
                      , float(arr[6])   # sqft_lot
                      , float(arr[7])   # floors
                      , float(arr[8])   # waterfront
                      , float(arr[9])   # view
                      , float(arr[10])  # condition
                      , float(arr[11])  # grade
                      , float(arr[12])  # sqft_above
                      , float(arr[13])  # sqft_basement
                      , float(arr[14])  # yr_built
                      , float(arr[15])  # yr_renovated
                      , float(arr[16])  # zipcode zip
                      , float(arr[17])  # lat
                      , float(arr[18])  # long
                      , float(arr[19])  # sqft_living15
                      , float(arr[20])  # sqft_lot15
                      ])

    if isNormalize:
        return [NormalizeData(x), y]
    else:
        return [x,y]
