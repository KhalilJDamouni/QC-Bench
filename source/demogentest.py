import tensorflow as tf
import glob
import sys
import torch
import json

if __name__ == "__main__":
    sspace = []
    folder = 'RESNET_CIFAR10'
    sspace = glob.glob(sys.path[0][0:-7]+"/models/DEMOGEN/ydjiang/experimental_results/model_dataset/"+folder+"/*")
    print("Subfolders: ", sspace)
    index=0
    with tf.compat.v1.Session() as sess:
        new_saver = tf.compat.v1.train.import_meta_graph(sspace[index] + "/model.ckpt-150000.meta")
        new_saver.restore(sess, sspace[index] + '/model.ckpt-150000')
        layers = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        print(len(layers))
        for layer in layers:
            print(layer.read_value())
    print(layers)
    tf.compat.v1.reset_default_graph()

   
    
    '''while (index<50):
        with tf.compat.v1.Session() as sess:
            #Load Model
            new_saver = tf.compat.v1.train.import_meta_graph(sspace[index] + "/model.ckpt-150000.meta")
            new_saver.restore(sess, sspace[index] + '/model.ckpt-150000')
            
            
            #Extract Weights
            variables_names = [v.name for v in tf.compat.v1.trainable_variables()]
            values = sess.run(variables_names)
            weights = []
            for k, v in zip(variables_names, values):
                if('conv' in k and "kernel" in k):
                    weights.append(torch.tensor(v.transpose((3,2,0,1))))
           
            layers = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            #Extract Performance
            with open(sspace[index] + "/eval.json", "r") as read_file:
                eval_data = json.load(read_file)
            with open(sspace[index] + "/train.json", "r") as read_file:
                train_data = json.load(read_file)
            print(len(layers))
            #print(len(weights))
            index +=1

    '''