import matplotlib
import pandas as pd
import numpy as np
import numpy.linalg as LA
from scipy import stats
import math
import numpy.ma as ma
import matplotlib.pyplot as plt
import numpy.ma as ma
from pathlib import Path
import sys

def agg(x, L, a=[]):
    if(L == 1):
        a = np.logical_not(ma.getmaskarray(x))
        return np.average(np.abs(x),weights=a,axis=1)
    if(L == 2):
        a = np.logical_not(ma.getmaskarray(x))
        return LA.norm(x,axis=1)/np.sqrt(np.sum(a,axis=1))
    if(L == 3):
        a = np.logical_not(ma.getmaskarray(x))
        #print(x[-1][-5:])
        #print(a[-1][-5:])
        #print(np.power(x,a/np.repeat(np.expand_dims(np.sum(a,axis=1),axis=1),a.shape[1],axis=1))[-1][-5:])
        return np.prod(np.power(x,a/np.repeat(np.expand_dims(np.sum(a,axis=1),axis=1),a.shape[1],axis=1)),axis=1)
    if(L == 4):
        a = a*np.logical_not(ma.getmaskarray(x))
        return np.average(np.abs(x),weights=a,axis=1)
    if(L == 5):
        a = a*np.logical_not(ma.getmaskarray(x))
        return np.prod(np.power(x,a/np.repeat(np.expand_dims(np.sum(a,axis=1),axis=1),a.shape[1],axis=1)),axis=1)

def all_aggs(in_chan, out_chan, in_weight, out_weight):
    return np.asarray([agg(np.ma.concatenate((in_chan,out_chan),axis=1),L=i,a=np.ma.concatenate((in_weight,out_weight),axis=1)) for i in range(1,6)])

if __name__ == "__main__":
    filename = "results-06-02-2021_10-24-29-NATSS-cifar10-90"
    file=Path(str(sys.path[0][0:-7])+"/outputs/"+filename+".csv")
    df = pd.read_csv(file,skip_blank_lines=False)
    data = dict()

    if(pd.isna(df.iloc[-1][1])):
        df = df.drop(labels=df.shape[0]-1, axis=0)

    for key in list(df.keys()):
        idx = list(np.where(pd.isna(df[key]))[0])
        idx = idx - np.arange(0,len(idx),1)
        data[key] = df[key].dropna(axis=0)
        data[key] = np.array_split(data[key],idx)
        maxLength = np.max(np.abs(np.diff(idx)))
        for i in range(len(data[key])):
            data[key][i] = np.append(data[key][i],(np.zeros((maxLength-len(data[key][i])))))
            shapes = data[key][0].shape
        temp = np.asarray(data[key])
        data[key] = ma.masked_array(temp, mask=(temp==0))

    data['in_QS_BE'] = np.arctan2(data['in_S_BE'],(1-1/data['in_C_BE']))
    data['out_QS_BE'] = np.arctan2(data['out_S_BE'],(1-1/data['out_C_BE']))
    data['in_QS_AE'] = np.arctan2(data['in_S_AE'],(1-1/data['in_C_AE']))
    data['out_QS_AE'] = np.arctan2(data['out_S_AE'],(1-1/data['out_C_AE']))

    aggregates = dict()

    aggregates['QS_BE'] = all_aggs(data['in_QS_BE'],data['out_QS_BE'],data['in_weight_BE'],data['out_weight_BE'])
    aggregates['QS_AE'] = all_aggs(data['in_QS_AE'],data['out_QS_AE'],data['in_weight_AE'],data['out_weight_AE'])
    aggregates['QE_BE'] = all_aggs(data['in_ER_BE'],data['out_ER_BE'],data['in_weight_BE'],data['out_weight_BE'])
    aggregates['QE_AE'] = all_aggs(data['in_ER_AE'],data['out_ER_AE'],data['in_weight_AE'],data['out_weight_AE'])
    aggregates['test_acc'] = np.mean(data['test_acc'],axis=1)
    aggregates['train_acc'] = np.mean(data['train_acc'],axis=1)
    aggregates['test_loss'] = np.mean(data['test_loss'],axis=1)
    aggregates['train_loss'] = np.mean(data['train_loss'],axis=1)
    aggregates['gap'] = np.mean(data['gap'],axis=1)

    #correlations
    X = ['test_acc','gap']
    Y = ['QS_BE','QS_AE','QE_BE','QE_AE']
    correlationsp = dict()
    correlationss = dict()
    for x in X:
        for y in Y:
            for i in range(5):
                correlationsp[y+'_'+x+'_L'+str(i+1)] = abs(stats.pearsonr(aggregates[x], aggregates[y][i])[0])
                correlationss[y+'_'+x+'_L'+str(i+1)] = abs(stats.spearmanr(aggregates[x], aggregates[y][i])[0])
    
    correlations = [correlationsp,correlationss]
    print(correlations)

    plt.subplot(2,1,1)
    plt.bar(correlationss.keys(),correlationss.values())
   
   
    plt.subplot(2,1,2)
    plt.plot(aggregates['QE_BE'][1],aggregates['test_acc'],'ro')
    plt.show()
