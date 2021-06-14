from torchvision import models
import torch

def VGGDepLists(model = models.vgg16):
	convIndexes = []
	for idx, param in enumerate(model.parameters()):
		tf = param.data
		if(tf.ndim == 4):
			convIndexes.append(idx)
		elif(tf.ndim == 2):
			break
	depLists = []
	for idx in range(len(convIndexes)-1):
		depLists.append([[convIndexes[idx], "out"], [convIndexes[idx+1], "in"]])
	return depLists
	
def ResNetAdj(model):
	convIdx = []
	downIdx = set()
	adjList = {}
	idx = 0
	for name, param in model.named_parameters():
		if(len(param.shape) == 4):
			convIdx.append(idx)
			if("downsample" in name):
				downIdx.add(idx)
		idx += 1

	idx = 0
	while(idx < len(convIdx)-1):
		if(convIdx[idx+1] in downIdx):
			adjList[convIdx[idx]] = [convIdx[idx+2]]
			adjList[convIdx[idx+1]] = [convIdx[idx+2]]
			idx += 2
		else:
			adjList[convIdx[idx]] = [convIdx[idx+1]]
			idx += 1
		
	idx = 0
	inc = 3
	nextinc = 3
	while(idx < len(convIdx)-4):
		inc = nextinc
		adjList[convIdx[idx]].append(convIdx[idx+inc])
		if((convIdx[idx+inc]) in downIdx):
			nextinc = 4
		else:
			nextinc = 3
		idx += inc-1
	return adjList, convIdx

def DepFromAdj(adjList, convIdx):
	depLists = []
	usedWhere = {}
	index = 0
	for key, values in adjList.items():
		alreadyUsed = False
		usedIndex = 0
		depList = []

		if(key in usedWhere):
			alreadyUsed = True 
			usedIndex = usedWhere[key]
		else:
			depList.append([key, 'out'])
			usedWhere[key] = index

		for idx in values: 
			if(idx + convIdx in usedWhere):
				alreadyUsed = True
				usedIndex = usedWhere[idx + convIdx]
			else:
				depList.append([idx, 'in'])
				usedWhere[idx + convIdx] = index

		if(alreadyUsed):
			for e in depList:
				if(e[1] == 'out'):
					usedWhere[e[0]] = usedIndex
				else:
					usedWhere[e[0] + convIdx] = usedIndex
			depLists[usedIndex].extend(depList)
		else:
			depLists.append(depList)
			index += 1

	#merge in out of the same layer together if necessary 
	for list in depLists:
		usedPlace = {}
		idx = 0
		while(idx < len(list)):
			if (list[idx][0] in usedPlace):
				usedIndex = list[idx][0]
				list.remove(list[idx])
				list[usedPlace[usedIndex]][1] = 'outin'
			else:
				usedPlace[list[idx][0]] = idx
				idx += 1

	return depLists

