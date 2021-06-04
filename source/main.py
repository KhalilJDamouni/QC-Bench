from parsing_agent import ParsingAgent
import numpy as np
import save
import json
import tensorflow as tf
import glob
import torch


if __name__ == "__main__":
    benchmark = 'zenNET' #from NATSS, NATST, NAS101, NAS201, DEMOGEN, NLP, zenNET
    dataset = 'ImageNet' #For NATs -> ImageNet16-120, cifar10, cifar100
                                #For DEMOGEN -> NIN_CIFAR10, RESNET_CIFAR10, RESNET_CIFAR100
                                #For zenNet -> CIFAR10, CIFAR100, ImageNet
    hp = '90'
    new = 1
    start = 0 #Seems to only work in batches of 35

    if(new):
        file_name = save.get_name(benchmark,dataset,hp)
    else:
        date = "06-02-2021_14-46-15"
        file_name = "outputs/results-"+date+"-"+benchmark+"-"+dataset+"-"+hp+".csv"

    counter = start

    agent = ParsingAgent(benchmark, dataset, hp, new, start)

    qualities, performance, laymod = agent.get_model()
    while (agent.index < len(agent.sspace)):
        if qualities.shape[0] != 0:

            print(str(agent.index)+'/'+str(len(agent.sspace)))

            performance = np.broadcast_to(performance,(qualities.shape[0],performance.shape[0]))
            to_write = np.concatenate((performance, qualities, laymod), axis=1)
            save.write(file_name,to_write)
            qualities, performance, laymod = agent.get_model()

  