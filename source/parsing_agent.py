import sys
import os
import tensorflow as tf
import process
import numpy as np
from nats_bench import create
import glob
import json
import torch


class ParsingAgent:
    index = 0
    api = None
    sspace = None

    def __init__(
            self,
            bench: str,
            dataset: str,
            hp: str,
            new: int,
            start: int):

        self.bench = bench
        self.dataset = dataset
        self.hp = hp

        if(bench == 'NATSS'):
            #download dataset?
            self.api = create(sys.path[0][0:-7]+'/models/NATSS', 'sss', fast_mode=True, verbose=False)
            self.sspace = glob.glob(sys.path[0][0:-7]+'/models/NATSS/*')

        if(bench == 'NATST'):
            #download dataset?
            self.api = create(sys.path[0][0:-7]+'/models/NATST', 'tss', fast_mode=True, verbose=False)
            self.sspace = glob.glob(sys.path[0][0:-7]+'/models/NATST/*')

        if(bench == 'DEMOGEN'):
            self.sspace = []
            self.sspace.extend(glob.glob("../models/DEMOGEN/demogen_models.tar/demogen_models/home/ydjiang/experimental_results/model_dataset/" + self.dataset + "/*"))
            print("Folders: ", self.sspace)


        if(bench == 'NLP'):
            self.sspace = glob.glob("../models/NAS-Bench-NLP/*")
            print('Folders: ', self.sspace)



        if(new != 1):
            self.index = start


    def get_model(self):
        #try:
        if(self.bench[0:-1] == 'NATS'):
            model_path = self.sspace[self.index]

            model_num = int((model_path.split(os.path.sep)[-1]).split('.')[0])

            weights = self.api.get_net_param(model_num, self.dataset, hp=self.hp, seed=None)
            weights = list((list(weights.values())[0]).values())
            weights = [weight for weight in weights if (len(weight.shape)==4)]

            performance = self.api.get_more_info(model_num, self.dataset, hp=self.hp, is_random=False)
            performance = [performance['test-accuracy']/100,performance['test-loss'],performance['train-accuracy']/100,performance['train-loss'],performance['test-accuracy']/100-performance['train-accuracy']/100]

            print(str(self.index)+"/"+str(len(self.sspace)))

        if(self.bench == 'DEMOGEN'):
            with tf.compat.v1.Session() as sess:
                #Get Model Number
                model_num = self.index

                #Load Model
                new_saver = tf.compat.v1.train.import_meta_graph(self.sspace[self.index] + "/model.ckpt-150000.meta")
                new_saver.restore(sess, self.sspace[self.index] + '/model.ckpt-150000')
                
                #Extract Weights
                variables_names = [v.name for v in tf.compat.v1.trainable_variables()]
                values = sess.run(variables_names)
                weights = []
                for k, v in zip(variables_names, values):
                    if('conv' in k and "kernel" in k):
                        weights.append(torch.tensor(v.transpose((3,2,0,1))))
                
                #Extract Performance
                with open(self.sspace[self.index] + "/eval.json", "r") as read_file:
                    eval_data = json.load(read_file)
                with open(self.sspace[self.index] + "/train.json", "r") as read_file:
                    train_data = json.load(read_file)
                
                performance = [eval_data["Accuracy"], eval_data["loss"], train_data["Accuracy"], train_data["loss"], eval_data["Accuracy"] - train_data["Accuracy"]]
            
        if(self.bench == 'NLP'):
            print("Model: ", self.index)

            #Get Weights
            weights = []
            model = torch.load(self.sspace[self.index])
            for name in model:
                if('raw' in name and len(model[name].shape) == 2):
                    print(name, model[name].shape)
                    weights.append(model[name])

            #Get Performance
            suffix = self.sspace[self.index].split('\\')[-1].replace("dump_weights_model_", "").replace('.pt',"")
            log = json.load(open('../nas-bench-nlp-release-master/train_logs_single_run/log_stats_model_' + suffix + '.json', 'r'))
            performance = [0, log['test_losses'][-1], 0, log['train_losses'][-1], 0]

        #except Exception as error:
        #    print(type(error))
        #    print(error)
        #    return None
            
        qualities, channel_weights = self.process_weights(weights)
        id = np.expand_dims(np.broadcast_to(model_num, len(channel_weights)),axis=1)
        laymod = np.concatenate((id,np.asarray(channel_weights)),axis=1)

        self.index+=1
        return np.asarray(qualities), np.asarray(performance), laymod

    def process_weights(self, weights):
        qualities = []
        channel_weights = []

        for weight in weights:
            layer_qualities, layer_weights = process.get_metrics(weight)
            qualities.append(layer_qualities)
            channel_weights.append(layer_weights)

        return qualities, channel_weights