from torchvision import models
from weightRelated import addNewWeights, Fold, extractWeights, transferWeights2d, transferWeightsHor2d, QC, cQC, Rank, cRank
from ScalingAlg import Greedy 
import torch
import numpy as np
from DepList import VGGDepLists, ResNetAdj, DepFromAdj

model = models.resnet34()

idx = 0
for name, param in model.named_parameters():
	print(idx, name, param.shape)
	idx += 1

adjList, convIdx = ResNetAdj(model)

print(adjList)

greatestIdx = convIdx[len(convIdx)-1]

depLists = DepFromAdj(adjList, greatestIdx)

print(depLists)



#depLists = getDepLists(model)

"""

#dictionairy inputs are the indexes of any conv layer
convWeights = {}
unfoldedConvWeights = {}
convSizes = {}

extractWeights(model, convWeights, unfoldedConvWeights, convSizes)

depLists = [[[0, 'out'], [2, 'in']], [[2, 'out'], [4, 'in']]]

Greedy(depLists, unfoldedConvWeights, convSizes)

Fold(convWeights, unfoldedConvWeights, convSizes)

addNewWeights(model, convWeights, convSizes)

for param in model.parameters():
	print(param.shape)

"""