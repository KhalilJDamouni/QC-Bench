from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras

import numpy as np

def create_model():
    model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)])

    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model

def get_checkpoint_path(root_dir=None):
    return os.path.join(self.get_model_dir_name(root_dir), CKPT_NAME)

if __name__ == '__main__':
    #root_dir = 'F:/Research/Multimedia/June/QC-Bench/QC-Bench/demogen/'
    #model_config = mc.ModelConfig(model_type='nin', dataset='cifar10', root_dir=root_dir)

    #model_path = get_checkpoint_path(root_dir)
    
    #model_config.load_parameters(model_path, sess)

    #manager = tf.train.CheckpointManager(ckpt, './../models/DEMOGEN/demogen_models.tar/demogen_models/home/ydjiang/experimental_results/model_dataset/NIN_CIFAR10/nin_wide_1.0x__dropout_0.0__decay_0.0_1', max_to_keep=3)

    # Define a simple sequential model

    # Create a basic model instance
    model = create_model()

    # Display the model's architecture
    model.summary()