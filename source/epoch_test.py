import pandas as pd
import numpy as np
import numpy.linalg as LA
import math
import numpy.ma as ma
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def get_headers(df):
    keys = list(df.keys())[1:]
    headers = list()
    for key in keys:
        key = '_'.join(key.split('_')[:-2]).lower()
        if key not in headers:
            headers.append(key)
        else:
            break
    assert(len(set(headers)) == len(headers))
    return headers

def norm(x, L, a=[]):
    #L: L1, or L2.
    if(L not in range(1,8)):
        print("Error: L must be 1:7")
        exit()
    if(L == 1):
        return np.mean(np.abs(x))
    if(L == 2):
        return LA.norm(x,axis=1)/math.sqrt(len(x[0,:]))
    if(L == 3):
        return np.prod(np.power(x,1/len(x[0,:])),axis=1)
    if(L == 4):
        #weighted average
        return np.average(np.abs(x),weights=a)
    if(L == 5):
        #weighted product
        return np.prod(np.power(x,a/(np.sum(a))),axis=1)
    if(L == 6):
        #weighted product with linear depth weights as well
        depth = np.arange(len(x))+1
        a = a*depth
        return np.prod(np.power(x,a/(np.sum(a))))
    if(L == 7):
        depth = np.flip(np.arange(len(x))+1)
        a = a*depth
        return np.prod(np.power(x,a/(np.sum(a))))

if __name__ == "__main__":
    f=Path(str(sys.path[0][0:-18])+"/outputs/results_date=2021-05-26-14-25-05_trial=0_ResNet34CIFAR_CIFAR10_Adasmomentum=0.9_weight_decay=0.0005_beta=0.98_linear=0.0_gamma=0.5_step_size=25.0_None_LR=0.03.xlsx")
    df = pd.read_excel(f)
    headers = get_headers(df)
    print(headers)
    df = df.T
    weights = np.ones(36)


    in_KG_BE = np.asarray(df.iloc[headers.index('in_s_be') + 1::len(headers),:])
    in_KG_BE = ma.masked_array(in_KG_BE, mask=in_KG_BE==0)
    out_KG_BE = np.asarray(df.iloc[headers.index('out_s_be') + 1::len(headers),:])
    out_KG_BE = ma.masked_array(out_KG_BE, mask=out_KG_BE==0)
    in_MC_BE = np.asarray(df.iloc[headers.index('in_condition_be') + 1::len(headers),:])
    in_MC_BE = ma.masked_array(in_MC_BE, mask=in_MC_BE==0)
    out_MC_BE = np.asarray(df.iloc[headers.index('out_condition_be') + 1::len(headers),:])
    out_MC_BE = ma.masked_array(out_MC_BE, mask=out_MC_BE==0)
    in_ER_BE = np.asarray(df.iloc[headers.index('in_er_be') + 1::len(headers),:])
    in_ER_BE = ma.masked_array(in_ER_BE, mask=in_ER_BE==0)
    out_ER_BE = np.asarray(df.iloc[headers.index('out_er_be') + 1::len(headers),:])
    out_ER_BE = ma.masked_array(out_ER_BE, mask=out_ER_BE==0)
    in_KG_AE = np.asarray(df.iloc[headers.index('in_s') + 1::len(headers),:])
    in_KG_AE = ma.masked_array(in_KG_AE, mask=in_KG_AE==0)
    out_KG_AE = np.asarray(df.iloc[headers.index('out_s') + 1::len(headers),:])
    out_KG_AE = ma.masked_array(out_KG_AE, mask=out_KG_AE==0)
    in_MC_AE = np.asarray(df.iloc[headers.index('in_condition') + 1::len(headers),:])
    in_MC_AE = ma.masked_array(in_MC_AE, mask=in_MC_AE==0)
    out_MC_AE = np.asarray(df.iloc[headers.index('out_condition') + 1::len(headers),:])
    out_MC_AE = ma.masked_array(out_MC_AE, mask=out_MC_AE==0)
    in_ER_AE = np.asarray(df.iloc[headers.index('in_er_ae') + 1::len(headers),:])
    in_ER_AE = ma.masked_array(in_ER_AE, mask=in_ER_AE==0)
    out_ER_AE = np.asarray(df.iloc[headers.index('out_er_ae') + 1::len(headers),:])
    out_ER_AE = ma.masked_array(out_ER_AE, mask=out_ER_AE==0)

    tag = 'test_acc1'
    if 'test_acc1' not in headers:
        tag = 'test_acc'
        if 'test_acc' not in headers:
            tag = 'acc'
    test_acc = np.asarray(
        df.iloc[headers.index(tag) + 1::len(headers), :])
    tag = 'train_acc1'
    if 'train_acc1' not in headers:
        tag = 'train_acc'
    try:
        train_acc = np.asarray(
            df.iloc[headers.index(tag) + 1::len(headers), :])
    except:
        print(1)
    train_loss = np.asarray(
                df.iloc[headers.index('train_loss') + 1::len(headers), :])
    try:
        test_loss = np.asarray(
                    df.iloc[headers.index('test_loss') + 1::len(headers), :])
    except:
        print(2)


    in_QG_BE = np.arctan2(in_KG_BE,(1-1/in_MC_BE))
    out_QG_BE = np.arctan2(out_KG_BE,(1-1/out_MC_BE))
    in_QG_AE = np.arctan2(in_KG_AE,(1-1/in_MC_AE))
    out_QG_AE = np.arctan2(out_KG_AE,(1-1/out_MC_AE))

    in_QG_BE = [norm(in_QG_BE,2),norm(in_QG_BE,3),norm(in_QG_BE,5,weights)]
    out_QG_BE = [norm(out_QG_BE,2),norm(out_QG_BE,3),norm(out_QG_BE,5,weights)]
    in_QG_AE = [norm(in_QG_AE,2),norm(in_QG_AE,3),norm(in_QG_AE,5,weights)]
    out_QG_BE = [norm(out_QG_AE,2),norm(out_QG_AE,3),norm(out_QG_AE,5,weights)]

    in_ER_BE = [norm(in_ER_BE,2),norm(in_ER_BE,3),norm(in_ER_BE,5,weights)]
    out_ER_BE = [norm(out_ER_BE,2),norm(out_ER_BE,3),norm(out_ER_BE,5,weights)]
    in_ER_AE = [norm(in_ER_AE,2),norm(in_ER_AE,3),norm(in_ER_AE,5,weights)]
    out_ER_AE = [norm(out_ER_AE,2),norm(out_ER_AE,3),norm(out_ER_AE,5,weights)]
