from __future__ import division
from source.parsing_agent import ParsingAgent
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

if __name__ == "__main__":
    benchmark = 'NATS' #from NATS, NAS101, NAS201, DEMOGEN
    dataset = 'Cifar10'
    hp = '90'
    new = 0
    start = 0

    if(new):
        file_name = save.get_name(benchmark,dataset,hp)
    else:
        date = ""
        file_name = "results-"+date+"-"+benchmark+"-"+dataset+"-"+hp+".csv"

    agent = ParsingAgent(benchmark, dataset, hp, new, start)

    qualities, performance = agent.get_model()
    while qualities != None:
        #write things?
        qualities, performance = agent.get_model()

  