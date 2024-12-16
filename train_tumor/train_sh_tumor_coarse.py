import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.system("python train_coarse_msd_renji.py")
os.system("python train_coarse_msd_rmyy.py")
os.system("python train_coarse_renji_rmyy.py")
