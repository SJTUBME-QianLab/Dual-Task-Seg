import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# --data denotes the testing data, which is not included in the training phase
os.system("python train_coarse.py --data renji")
os.system("python train_coarse.py --data msd")
os.system("python train_coarse.py --data rmyy")
