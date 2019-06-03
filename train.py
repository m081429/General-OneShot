from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os, sys
import tensorflow as tf

from preprocess import Preprocess
from data_runner import DataRunner
from model_factory import compile_model
from callbacks import CallBacks

tf.config.gpu.set_per_process_memory_growth(True)

###############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(description='Run a Siamese Network with a triplet loss on a folder of images.')
parser.add_argument("-t", "--image_dir_train",
                    dest='image_dir_train',
                    required=True,
                    help="File path ending in folders that are to be used for model training")

parser.add_argument("-v", "--image_dir_validation",
                    dest='image_dir_validation',
                    default=None,
                    help="File path ending in folders that are to be used for model validation")


parser.add_argument("-m", "--model-name",
                    dest='model_name',
                    default='VGG16',
                    choices=['DenseNet121',
                             'DenseNet169',
                             'DenseNet201',
                             'InceptionResNetV2',
                             'InceptionV3',
                             'MobileNet',
                             'MobileNetV2',
                             'NASNetLarge',
                             'NASNetMobile',
                             'ResNet50',
                             'VGG16',
                             'VGG19',
                             'Xception'],
                    help="Models available from tf.keras")

parser.add_argument("-o", "--optimizer-name",
                    dest='optimizer',
                    default='Adam',
                    choices=['Adadelta',
                             'Adagrad',
                             'Adam',
                             'Adamax',
                             'Ftrl',
                             'Nadam',
                             'RMSprop',
                             'SGD'],
                    help="Optimizers from tf.keras")


parser.add_argument("-p", "--patch_size",
                    dest='patch_size',
                    help="Patch size to use for training",
                    default=256, type=int)

parser.add_argument("-l", "--log_dir",
                    dest='log_dir',
                    default='log_dir',
                    help="Place to store the tensorboard logs")

parser.add_argument("-r", "--learning-rate",
                    dest='lr',
                    help="Learning rate",
                    default=0.0001, type=float)

parser.add_argument("-e", "--num_epocs",
                    dest='num_epochs',
                    help="Number of epochs to use for training",
                    default=10, type=int)

parser.add_argument("-b", "--batch-size",
                    dest='BATCH_SIZE',
                    help="Number of batches to use for training",
                    default=1, type=int)

parser.add_argument("-w", "--num-workers",
                    dest='NUM_WORKERS',
                    help="Number of workers to use for training",
                    default=10, type=int)

parser.add_argument("-s", "--steps-per-epoch",
                    dest='spe',
                    help="Limit number of examples for faster iterating",
                    default=None, type=int)

parser.add_argument("-m", "--use-multiprocessing",
                    dest='use_multiprocessing',
                    help="Whether or not to use multiprocessing",
                    default=True, type=bool)



parser.add_argument("-V", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="INFO",
                    help="Set the logging level")

args = parser.parse_args()

logging.basicConfig(stream=sys.stderr, level=args.logLevel,
                        format='%(name)s (%(levelname)s): %(message)s')




###############################################################################
# Begin priming the data generation pipeline
###############################################################################

# Get Training and Validation data
train_data = Preprocess(args.image_dir_train)

train_ds = DataRunner(train_data.set_list
                      train=True,
                      image_size=args.patch_size)

ds_t = tf.data.Dataset.from_generator(train_ds.get_distributed_datasets(), output_types=(
    {
        "anchor": tf.float32,
        "pos_img": tf.float32,
        "neg_img": tf.float32
    }, tf.int64)).batch(args.BATCH_SIZE).repeat()


if args.image_dir_validation:
    val_data = Preprocess(image_dir_validation)
    val_ds = DataRunner(val_data.set_list,
                        train=False,
                        image_size=args.patch_size)
    ds_v = tf.data.Dataset.from_generator(val_ds.get_distributed_datasets(), output_types=(
        {
            "anchor": tf.float32,
            "pos_img": tf.float32,
            "neg_img": tf.float32
        }, tf.int64)).batch(args.BATCH_SIZE).repeat()
    validation_steps = 1000
else:
    ds_v = None
    validation_steps = None

# I now have generators for training and validation

###############################################################################
# Build the model
###############################################################################
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    model = compile_model(args.optimizer, args.lr, img_size=args.patch_size)

###############################################################################
# Define callbacks
###############################################################################
cb = CallBacks(learning_rate=args.lr, log_dir=args.log_dir, optimizer=args.optimizer)

###############################################################################
# Run the training
###############################################################################
model.fit(ds_t,
          steps_per_epoch=train_ds.image_file_list.__len__() / args.BATCH_SIZE,
          epochs=args.num_epochs,
          callbacks=cb,
          validation_data=ds_v,
          validation_steps=100,
          class_weight=None,
          max_queue_size=1000,
          workers=args.NUM_WORKERS,
          use_multiprocessing=args.use_multiprocessing,
          shuffle=False,
          initial_epoch=0
          )