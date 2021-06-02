from parsing_agent import ParsingAgent
import numpy as np
import save
import json
import tensorflow as tf
import glob
import torch


if __name__ == "__main__":
    benchmark = 'NATSS' #from NATSS, NATST, NAS101, NAS201, DEMOGEN
    dataset = 'cifar10' #For NATs -> ImageNet16-120, cifar10, cifar100
    hp = '90'
    new = 1
    start = 0

    if(new):
        file_name = save.get_name(benchmark,dataset,hp)
    else:
        date = ""
        file_name = "outputs/results-"+date+"-"+benchmark+"-"+dataset+"-"+hp+".csv"

    agent = ParsingAgent(benchmark, dataset, hp, new, start)

    qualities, performance, laymod = agent.get_model()
    while type(qualities)!=type(None):
        if qualities.shape[0] != 0:
            performance = np.broadcast_to(performance,(qualities.shape[0],performance.shape[0]))
            to_write = np.concatenate((performance, qualities, laymod), axis=1)
            save.write(file_name,to_write)
            qualities, performance, laymod = agent.get_model()

  