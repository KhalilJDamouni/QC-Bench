import csv
import pandas as pd
import numpy as np
import glob
import qualities
import plotting_func
import matplotlib.pyplot as plt

def plot(path, c):
    #path_new = path + "LilJon-Adaptive-" + c + ""
    files = glob.glob(path)
    print(files)
    print(len(files))

    results = {} # [epochs][metrics]
    results
    for epoch in range(70):
        file = [f for f in files if "-" + str(epoch) + ".csv" in f]
        file = file[0]
        print(epoch, end=' ')
        results[epoch] = qualities.correlate(file);

    plottingBE = {}
    plottingBE['gap']  = {}
    plottingBE['test'] = {}

    plottingAE = {}
    plottingAE['gap']  = {}
    plottingAE['test'] = {}

    #BE and Test
    for x in results[epoch]['spearman']:
        if(x == 'QE_BE_test_acc_L2_6'): 
            plottingBE['test'][x] = {}
        if(x == 'QS_BE_test_acc_L3_0'):
            plottingBE['test'][x] = {}
        if(x == 'spec_BE_test_acc_L3_7'):
            plottingBE['test'][x] = {}
        if(x == 'fro_BE_test_acc_L3_7'):
            plottingBE['test'][x] = {}
        #Can add gap if wanted

    #BE and Gap
    for x in results[epoch]['spearman']:
        if(x == 'QE_BE_gap_L2_6'): 
            plottingBE['gap'][x] = {}
        if(x == 'QS_BE_gap_L3_0'):
            plottingBE['gap'][x] = {}
        if(x == 'spec_BE_gap_L3_7'):
            plottingBE['gap'][x] = {}
        if(x == 'fro_BE_gap_L3_7'):
            plottingBE['gap'][x] = {}
        #Can add gap if wanted

    #AE and Test
    for x in results[epoch]['spearman']:
        if(x == 'QE_AE_test_acc_L2_6'):
            plottingAE['test'][x] = {}
        if(x == 'QS_AE_test_acc_L3_0'):
            plottingAE['test'][x] = {}
        if(x == 'spec_AE_test_acc_L3_7'):
            plottingAE['test'][x] = {}
        if(x == 'fro_AE_test_acc_L3_7'):
            plottingAE['test'][x] = {}
    #AE and Gap
    for x in results[epoch]['spearman']:
        if(x == 'QE_AE_gap_L2_6'): 
            plottingAE['gap'][x] = {}
        if(x == 'QS_AE_gap_L3_0'):
            plottingAE['gap'][x] = {}
        if(x == 'spec_AE_gap_L3_7'):
            plottingAE['gap'][x] = {}
        if(x == 'fro_AE_gap_L3_7'):
            plottingAE['gap'][x] = {}
            
    for epoch in range(70):
        for x in plottingBE['test']:
            plottingBE['test'][x][epoch] = results[epoch]['spearman'][x]
        for x in plottingAE['test']:
            plottingAE['test'][x][epoch] = results[epoch]['spearman'][x]
        for x in plottingBE['gap']:
            plottingBE['gap'][x][epoch] = results[epoch]['spearman'][x]
        for x in plottingAE['gap']:
            plottingAE['gap'][x][epoch] = results[epoch]['spearman'][x]

    value_list = [plottingAE['test'], plottingBE['test'], plottingAE['gap'], plottingBE['gap']]
    title_list = ['Correlation with Test Acc. with LRF',
                'Correlation with Test Acc. without LRF',
                'Correlation with Gen. Gap with LRF',
                'Correlation with Gen. Gap without LRF']

    COLORS2 = [[0.89803922, 0.49803922, 1.], [0.47058824, 0.2745098 , 1.], [0.50980392, 0, 0],[0.98431373, 0.48627451, 0.04705882]]

    order = [['QE_AE_test_acc_L2_6', 'QS_AE_test_acc_L3_0', 'spec_AE_test_acc_L3_7', 'fro_AE_test_acc_L3_7'],
             ['QE_BE_test_acc_L2_6', 'QS_BE_test_acc_L3_0', 'spec_BE_test_acc_L3_7', 'fro_BE_test_acc_L3_7'],
             ['QE_AE_gap_L2_6', 'QS_AE_gap_L3_0', 'spec_AE_gap_L3_7', 'fro_AE_gap_L3_7'],
             ['QE_BE_gap_L2_6', 'QS_BE_gap_L3_0', 'spec_BE_gap_L3_7', 'fro_BE_gap_L3_7']]

    #dict_keys(['QS_AE_test_acc_L3_0', 'QE_AE_test_acc_L2_6', 'spec_AE_test_acc_L3_7', 'fro_AE_test_acc_L3_7'])
    #dict_keys(['QS_BE_test_acc_L3_0', 'QE_BE_test_acc_L2_6', 'spec_BE_test_acc_L3_7', 'fro_BE_test_acc_L3_7'])
    #dict_keys(['QS_AE_gap_L3_0', 'QE_AE_gap_L2_6', 'spec_AE_gap_L3_7', 'fro_AE_gap_L3_7'])
    #dict_keys(['QS_BE_gap_L3_0', 'QE_BE_gap_L2_6', 'spec_BE_gap_L3_7', 'fro_BE_gap_L3_7'])


    #Label Sizes
    title_size = 42
    x_label_size = 42
    y_label_size = 42
    x_tick_size = 35
    y_tick_size = 35

    for p in range(4):
        ax_list = []
        ln_list = []
        fig, ax1 = plt.subplots()
        fig.set_size_inches(15,10)

        plt.plot(value_list[p][order[p][0]].keys(), 
                    value_list[p][order[p][0]].values(),
                    color=COLORS2[0],lw=4)
        #0
        ax1.set_xlabel('Epoch - (t)',fontsize=x_label_size)
        #ax1.set_yticks([])
        #ax1.set_ylim([0, 1])
        ax1.tick_params(axis = 'y', labelsize = y_tick_size)
        ax1.tick_params(axis='x',labelsize=x_tick_size)
        
        #1
        plt.plot(value_list[p][order[p][1]].keys(), 
                        value_list[p][order[p][1]].values(),
                        color=COLORS2[1],lw=4)

        #2
        plt.plot(value_list[p][order[p][2]].keys(), 
                        value_list[p][order[p][2]].values(),
                        color=COLORS2[2],lw=4)

        #3
        plt.plot(value_list[p][order[p][3]].keys(), 
                        value_list[p][order[p][3]].values(),
                        color=COLORS2[3],lw=4)

        plt.title(title_list[p], fontsize = title_size)
        plt.xlabel('Epoch')
        ax1.set_ylabel('Spearman Correlation', fontsize = x_label_size)
        #plt.legend(legend_list[p])
        if('Gap' in title_list[p]):
            plt.ylim([-1, 1])
        else:
            plt.ylim([-1, 1])
        plt.savefig('plotting/figures/adaptive_'+ title_list[p] + '.png', bbox_inches = 'tight')
        #plt.show()


if __name__ == "__main__":
    path = "plotting/csv_files/New/Adaptives/Combined/*"
    plot(path, 'CIFAR10')

    # exit()
    # ops = ['AdaM', 'RMSGD', 'SAM', 'SGD']
    # cs  = ['CIFAR10', 'CIFAR100']
    # for op in ops:
    #     for c in cs:
    #         path = "plotting/csv_files/New/Adaptives/"
    #         print(path)
    #         plot(path)

