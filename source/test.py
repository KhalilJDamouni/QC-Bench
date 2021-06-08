from __future__ import division
import parsing_agent
import sys
import random
import torch
import glob
import os
import math
import main

import resnet_cifar
import numpy.linalg as LA
import numpy as np
import torch
from nats_bench import create
from scipy.optimize import minimize_scalar
from pprint import pprint

model = None
model = torch.load(str(sys.path[0][0:-7])+'/source/best.pth.tar')

weights = []

for name in model['state_dict_network']:
    if ("conv" in name or "fc" in name) and "weight" in name:
        weights.append(model['state_dict_network'][name])

slave = parsing_agent.ParsingAgent("NATSS", "cifar10", '90', 1, 0)
print(slave.process_weights(weights))

