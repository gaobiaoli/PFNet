import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
if __name__=="__main__":
    data=pd.read_csv("F:\code_python\PFNet-pytorch\loss\PFnet.csv")
    print(1)
    tx=data.iloc[:,0]
    ty=data.iloc[:,1]
    vx=data.iloc[:,2]
    vy=data.iloc[:,3]
    plt.figure()
    plt.plot(tx,ty,color='#cba0e6',label="Train Loss")
    plt.plot(vx,vy,color='#FFB177',label="Val Loss",LineStyle="--")
    plt.legend(loc="upper right")
    plt.xlabel('Loss')
    plt.ylabel('Epoch1')
    plt.savefig("loss.png")