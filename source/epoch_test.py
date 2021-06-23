import pandas as pd
import numpy as np
import numpy.linalg as LA
import math
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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
    f=Path(str(sys.path[0][0:-7])+"/outputs/epochcifar100.xlsx")
    df = pd.read_excel(f)
    headers = get_headers(df)
    print(headers)
    df = df.T
    weights = np.array([3,64,64,64,64,64,64,64,128,64,128,128,128,128,128,128,128,256,128,256,256,256,256,
    256,256,256,256,256,256,256,512,256,512,512,512,512,27,64,64,64,64,64,64,128,128,128,128,128,128,
    128,128,128,256,256,256,256,256,256,256,256,256,256,256,256,256,512,512,512,512,512,512,512])

    in_KG_BE = np.asarray(df.iloc[headers.index('in_s_be') + 1::len(headers),:])
    in_KG_BE = ma.masked_array(in_KG_BE, mask=in_KG_BE==0)
    out_KG_BE = np.asarray(df.iloc[headers.index('out_s_be') + 1::len(headers),:])
    out_KG_BE = ma.masked_array(out_KG_BE, mask=out_KG_BE==0)
    KG_BE = np.ma.concatenate((in_KG_BE,out_KG_BE),axis=1)
    in_MC_BE = np.asarray(df.iloc[headers.index('in_condition_be') + 1::len(headers),:])
    in_MC_BE = ma.masked_array(in_MC_BE, mask=in_MC_BE==0)
    out_MC_BE = np.asarray(df.iloc[headers.index('out_condition_be') + 1::len(headers),:])
    out_MC_BE = ma.masked_array(out_MC_BE, mask=out_MC_BE==0)
    MC_BE = np.ma.concatenate((in_MC_BE,out_MC_BE),axis=1)
    in_ER_BE = np.asarray(df.iloc[headers.index('in_er_be') + 1::len(headers),:])
    in_ER_BE = ma.masked_array(in_ER_BE, mask=in_ER_BE==0)
    out_ER_BE = np.asarray(df.iloc[headers.index('out_er_be') + 1::len(headers),:])
    out_ER_BE = ma.masked_array(out_ER_BE, mask=out_ER_BE==0)
    ER_BE = np.ma.concatenate((in_ER_BE,out_ER_BE),axis=1)
    in_KG_AE = np.asarray(df.iloc[headers.index('in_s') + 1::len(headers),:])
    in_KG_AE = ma.masked_array(in_KG_AE, mask=in_KG_AE==0)
    out_KG_AE = np.asarray(df.iloc[headers.index('out_s') + 1::len(headers),:])
    out_KG_AE = ma.masked_array(out_KG_AE, mask=out_KG_AE==0)
    KG_AE = np.ma.concatenate((in_KG_AE,out_KG_AE),axis=1)
    in_MC_AE = np.asarray(df.iloc[headers.index('in_condition') + 1::len(headers),:])
    in_MC_AE = ma.masked_array(in_MC_AE, mask=in_MC_AE==0)
    out_MC_AE = np.asarray(df.iloc[headers.index('out_condition') + 1::len(headers),:])
    out_MC_AE = ma.masked_array(out_MC_AE, mask=out_MC_AE==0)
    MC_AE = np.ma.concatenate((in_MC_AE,out_MC_AE),axis=1)
    in_ER_AE = np.asarray(df.iloc[headers.index('in_er_ae') + 1::len(headers),:])
    in_ER_AE = ma.masked_array(in_ER_AE, mask=in_ER_AE==0)
    out_ER_AE = np.asarray(df.iloc[headers.index('out_er_ae') + 1::len(headers),:])
    out_ER_AE = ma.masked_array(out_ER_AE, mask=out_ER_AE==0)
    ER_AE = np.ma.concatenate((in_ER_AE,out_ER_AE),axis=1)
    in_fro_BE = np.asarray(df.iloc[headers.index('in_fro_be') + 1::len(headers),:])
    in_fro_BE = ma.masked_array(in_fro_BE, mask=in_fro_BE==0)
    out_fro_BE = np.asarray(df.iloc[headers.index('out_fro_be') + 1::len(headers),:])
    out_fro_BE = ma.masked_array(out_fro_BE, mask=out_fro_BE==0)
    fro_BE = np.ma.concatenate((in_fro_BE,out_fro_BE),axis=1)
    in_spec_BE = np.asarray(df.iloc[headers.index('in_spec_be') + 1::len(headers),:])
    in_spec_BE = ma.masked_array(in_spec_BE, mask=in_spec_BE==0)
    out_spec_BE = np.asarray(df.iloc[headers.index('out_spec_be') + 1::len(headers),:])
    out_spec_BE = ma.masked_array(out_spec_BE, mask=out_spec_BE==0)
    spec_BE = np.ma.concatenate((in_spec_BE,out_spec_BE),axis=1)
    in_fro_AE = np.asarray(df.iloc[headers.index('in_fro_ae') + 1::len(headers),:])
    in_fro_AE = ma.masked_array(in_fro_AE, mask=in_fro_AE==0)
    out_fro_AE = np.asarray(df.iloc[headers.index('out_fro_ae') + 1::len(headers),:])
    out_fro_AE = ma.masked_array(out_fro_AE, mask=out_fro_AE==0)
    fro_AE = np.ma.concatenate((in_fro_AE,out_fro_AE),axis=1)
    in_spec_AE = np.asarray(df.iloc[headers.index('in_spec_ae') + 1::len(headers),:])
    in_spec_AE = ma.masked_array(in_spec_AE, mask=in_spec_AE==0)
    out_spec_AE = np.asarray(df.iloc[headers.index('out_spec_ae') + 1::len(headers),:])
    out_spec_AE = ma.masked_array(out_spec_AE, mask=out_spec_AE==0)
    spec_AE = np.ma.concatenate((in_spec_AE,out_spec_AE),axis=1)
    in_noise = np.asarray(df.iloc[headers.index('in_noise') + 1::len(headers),:])
    in_noise = ma.masked_array(in_noise, mask=in_noise==0)
    out_noise = np.asarray(df.iloc[headers.index('out_noise') + 1::len(headers),:])
    out_noise = ma.masked_array(out_noise, mask=out_noise==0)
    noise = np.ma.concatenate((in_noise,out_noise),axis=1)

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


    gap = train_acc - test_acc

    QG_BE = np.arctan2(KG_BE,(1-1/MC_BE))
    QG_AE = np.arctan2(KG_AE,(1-1/MC_AE))

    QG_BE = np.asarray([norm(QG_BE,2),norm(QG_BE,3)])
    QG_AE = np.asarray([norm(QG_AE,2),norm(QG_AE,3)])
    ER_BE = np.asarray([norm(ER_BE,2),norm(ER_BE,3)])
    ER_AE = np.asarray([norm(ER_AE,2),norm(ER_AE,3)])

    x=np.arange(0,250,1)

    fig, ax1 = plt.subplots()

    ln1 = ax1.plot(x, test_acc,color='blue')
    ax1.tick_params(axis='y')
    ax1.set_ylabel('test_accuracy')
    ax1.set_xlabel('epoch')

    ax2 = ax1.twinx()
    ln2 = ax2.plot(x, QG_AE[1,:],color='red')
    ax2.set_yticks([])

    ax3 = ax1.twinx()
    ln3 = ax3.plot(x, QG_BE[1,:],color='green')
    ax3.set_yticks([])

    ax4 = ax1.twinx()
    ln4 = ax4.plot(x, ER_BE[0,:],color='orange')
    ax4.set_yticks([])

    ax5 = ax1.twinx()
    ln5 = ax5.plot(x, ER_AE[0,:],color='purple')
    ax5.set_yticks([])

    legend_elements = [Line2D([0], [0], color='b', lw=2, label='test accuracy'),Line2D([0], [0], color='r', lw=2, label=r'$\widehat{Q}_{G}^{p}$'),Line2D([0], [0], color='g', lw=2, label=r'${Q}_{G}^{p}$'),
    Line2D([0], [0], color='orange', lw=2, label=r'${Q}_{ER}^{L2}$'),Line2D([0], [0], color='purple', lw=2, label=r'$\widehat{Q}_{ER}^{L2}$')]
    ax1.legend(handles=legend_elements, loc=4)

    fig.tight_layout()
    plt.show()
