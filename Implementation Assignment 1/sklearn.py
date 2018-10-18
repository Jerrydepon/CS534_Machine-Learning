import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

aplha = 0.01 #learning rate

dataset = pd.read_csv('PA1_train.csv')
x_train = dataset.iloc[:, 3:-1]
print(x_train)
y_train = dataset.iloc[:, -1]
#print(y_train)
dataset = pd.read_csv('PA1_test.csv')
x_test = dataset.iloc[:, 3:]
#print(x_test)
y_test = dataset.iloc[:, -1]
#print(y_test)
dataset = pd.read_csv('PA1_dev.csv')
x_test2 = dataset.iloc[:, 3:-1]
#print(x_test2)]
y_test2 = dataset.iloc[:, -1]
#print(y_test2)

# With the method from Sklearn
# =============================================================================
# split the dataset into train & test of necessary
# from sklearn.cross_validation import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
 
## apply linear regression based on training examples
#regressor = LinearRegression()
#regressor.fit(x_train, y_train)
## see the values of parameters after training
#print("constant parameter is", regressor.intercept_)
#for i, j in enumerate(x_train.columns):
#    print("parameter of {} is {}".format(j, regressor.coef_[0])) 
# 
## predict the result of test examples
#y_pred = regressor.predict(x_test2)
#print("y predict: ", y_pred)
# 
## measuring the accuracy
#print("Mean Accuracy: ",regressor.score(x_train, y_train))
#print("MSE: ", mean_squared_error(y_pred, y_test2))
# 
### Plot the graph
##plt.scatter()
##plt.title("TBD")
##plt.show()

# =============================================================================
# Alogorithm Model
# =============================================================================
num_row = x_train.shape[0]
num_col = x_train.shape[1]

b = 3 # initial b 
w = np.full(x_train.shape[1], 3) # initial w
j = np.zeros(x_train.shape[1])
limit = 0.5 # the norm of the gradient of convergence
#print(w)
#print(j)

# =============================================================================
# Cost Function
# =============================================================================
diff = 0
summ = 0
for i in range(num_row):
    h_theta = b + 0 # hypothesis
    for j in range(num_col):
        h_theta = h_theta * x_train.iloc[i,j] # h_theta(x)
    diff = (h_theta - y_test[i])**2
    summ = summ + diff
sse = 1 / (2 * num_row) * summ
print("SSE: ", sse)

# =============================================================================
# Gradient Descent
# =============================================================================

        
   
#b_grad = 0.0
#w_grad = 0.0        