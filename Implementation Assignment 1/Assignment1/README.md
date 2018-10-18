Python version using - python3.7
 
1.  cd /scratch/cs534
2.  /bin/bash
get in Bash
3.  source myenv3.5/bin/activate
source python env
4.  cd {path}/ 
cd path to the place of pa1.py 
5.  vim pa1.py
Setting the input value
alphaVal = 10 ** (-5)               # learning rate
limit = 0.5                         # convergence condition
maxIter = 10000                     # limitation of iteration
lam = 0.0                           # regularization coefficient
outPutFile = "pa1_result_"          # Out put file name
isValidate = True                   # is Out put validation result
isNormalize = True                  # is Normalize input date
trainingFile = "PA1_train.csv"      # Training file name
ValidateFile = "PA1_dev.csv"        # Validate file name
6.  python pa1.py
7.  ls
the path will out csv files for training and validation result detail, which include the SSE and the norm in each iteration.
