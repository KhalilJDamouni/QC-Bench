from __future__ import division
import sys
import random
from numpy.core.function_base import linspace
import torch
import glob
import os
import math
import process
import correlate
import save
import numpy.linalg as LA
import numpy as np
from nats_bench import create
from scipy.optimize import minimize_scalar
from pprint import pprint

def norm(x, L, a=[]):
    #L: L1, or L2.
    if(L not in range(1,8)):
        print("Error: L must be 1:7")
        exit()
    if(L == 1):
        return np.mean(np.abs(x))
    if(L == 2):
        return LA.norm(x)/math.sqrt(len(x))
    if(L == 3):
        return np.prod(np.power(x,1/len(x)))
    if(L == 4):
        #weighted average
        return np.average(np.abs(x),weights=a)
    if(L == 5):
        #weighted product
        return np.prod(np.power(x,a/(np.sum(a))))
    if(L == 6):
        #weighted product with linear depth weights as well
        depth = np.arange(len(x))+1
        a = a*depth
        return np.prod(np.power(x,a/(np.sum(a))))
    if(L == 7):
        depth = np.flip(np.arange(len(x))+1)
        a = a*depth
        return np.prod(np.power(x,a/(np.sum(a))))



def get_quality(model_params):
    ER_BE_list = []
    ER_AE_list = []
    mquality_BE_list = []
    mquality_AE_list = []
    rank_AE_list = []
    weight_ER_BE_list = []
    weight_ER_AE_list = []
    weight_mq_BE_list = []
    weight_mq_AE_list = []
    weight_rank_AE_list = []
    for i in model_params:
        for k, v in (model_params[i]).items():
            if('weight' in k and len(list(v.size())) == 4 and v.shape[3]!=1):
                #print(k)
                #print("\n")
                mquality_BE, mquality_AE, ER_BE, ER_AE, weight_BE, weight_AE, rank_AE = process.get_metrics(model_params,i,k)
                if(mquality_BE[0]>0):
                    mquality_BE_list.append(mquality_BE[0])
                    weight_mq_BE_list.append(weight_BE[0])
                if(mquality_BE[1]>0):
                    mquality_BE_list.append(mquality_BE[1])
                    weight_mq_BE_list.append(weight_BE[1])
                if(mquality_AE[0]>0):
                    mquality_AE_list.append(mquality_AE[0])
                    weight_mq_AE_list.append(weight_AE[0])
                if(mquality_AE[1]>0):
                    mquality_AE_list.append(mquality_AE[1])
                    weight_mq_AE_list.append(weight_AE[1])
                if(ER_BE[0]>0):
                    ER_BE_list.append(ER_BE[0])
                    weight_ER_BE_list.append(weight_BE[0])
                if(ER_BE[1]>0):
                    ER_BE_list.append(ER_BE[1])
                    weight_ER_BE_list.append(weight_BE[1])
                if(ER_AE[0]>0):
                    ER_AE_list.append(ER_AE[0])
                    weight_ER_AE_list.append(weight_AE[0])
                if(ER_AE[1]>0):
                    ER_AE_list.append(ER_AE[1])
                    weight_ER_AE_list.append(weight_AE[1])
                if(rank_AE[0]>0):
                    rank_AE_list.append(rank_AE[0])
                    weight_rank_AE_list.append(weight_AE[0])
                if(rank_AE[1]>0):
                    rank_AE_list.append(rank_AE[1])
                    weight_rank_AE_list.append(weight_AE[1])
    if(len(mquality_BE_list)==0):
        print("empty BE")
        return None
    else:
        return [norm(mquality_BE_list,1),norm(mquality_BE_list,2),norm(mquality_BE_list,3),norm(mquality_BE_list,4,weight_mq_BE_list),norm(mquality_BE_list,5,weight_mq_BE_list),
norm(mquality_AE_list,1),norm(mquality_AE_list,2),norm(mquality_AE_list,3),norm(mquality_AE_list,4,weight_mq_AE_list),norm(mquality_AE_list,5,weight_mq_AE_list),        
norm(ER_BE_list,1),norm(ER_BE_list,2),norm(ER_BE_list,3),norm(ER_BE_list,4,weight_ER_BE_list),norm(ER_BE_list,5,weight_ER_BE_list),
norm(ER_AE_list,1),norm(ER_AE_list,2),norm(ER_AE_list,3),norm(ER_AE_list,4,weight_ER_AE_list),norm(ER_AE_list,5,weight_ER_AE_list),
norm(rank_AE_list,1),norm(rank_AE_list,2),norm(rank_AE_list,3),norm(rank_AE_list,4,weight_rank_AE_list),norm(rank_AE_list,5,weight_rank_AE_list)]


if __name__ == "__main__":
    searchspace = 'sss'
    api = create(sys.path[0][0:-7]+'/fake_torch_dir/models'+searchspace[0], searchspace, fast_mode=True, verbose=False)
    dataset = 'ImageNet16-120'
    hp = '12'
    early_stop=100000
    i=0
    new = 0

    pickles=glob.glob(sys.path[0][0:-7]+'/fake_torch_dir/models'+searchspace[0]+'/*')
    #model_qualities = []
    #test_accuracy = []

    if(new):
        file_name = save.get_name()
    else:
        file_name = "outputs/" + "imagenete12" + ".csv"
        lastmodel = 31010
    '''
    params = api.get_net_param(11197, dataset, None)
    model_val = get_quality(params)

    '''
    for model in pickles:
        if((model.split(os.path.sep)[-1]).split('.')[0]=='meta'):
            print("skipping meta")
            continue
        if(not new):
            i+=1
            early_stop+=1
            model_num = int((model.split(os.path.sep)[-1]).split('.')[0])
            if(model_num==lastmodel):
                new = 1
            continue
        model_vals = []
        if(i+1>early_stop):
            break
        
        model_num = int((model.split(os.path.sep)[-1]).split('.')[0])
        print(str(i+1)+'/'+str(early_stop))
        print("model: "+str(model_num))
        params = api.get_net_param(model_num, dataset, hp=hp, seed=None)
        model_val = get_quality(params)
        if(model_val):
            model_vals.append(model_num)
            info = api.get_more_info(model_num, dataset, hp=hp, is_random=False)
            #test_accuracy.append(info['test-accuracy']/100)
            #print(info)
            model_vals.append(info['test-accuracy']/100)
            model_vals.append(info['test-loss'])
            model_vals.append(info['train-accuracy']/100)
            model_vals.append(info['train-loss'])
            model_vals.append((info['test-accuracy']-info['train-accuracy'])/100)
            #model_qualities.append(get_quality(params))
            model_vals.extend(model_val)
            #print(model_vals)
            save.write(file_name,model_vals)
        else:
            print("skipping 0 model")

            
        print("\n")
        i+=1



    '''
    print(str(model_qualities),str(test_accuracy))

    p_corr = correlate.pearson_corr(model_qualities, test_accuracy)
    ro_corr = correlate.rank_order_corr(model_qualities, test_accuracy)
    
    print(p_corr,ro_corr)

    correlate.display(model_qualities, test_accuracy)
    '''
