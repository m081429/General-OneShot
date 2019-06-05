from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os, sys
import tensorflow as tf

from preprocess import Preprocess
from data_runner import DataRunner
from model_factory import GetModel
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

parser.add_argument("-e", "--num-epochs",
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
                    default=5, type=int)

parser.add_argument("--use-multiprocessing",
                    help="Whether or not to use multiprocessing",
                    const=True, default=True, nargs='?',
                    type=bool)

parser.add_argument("-V", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="INFO",
                    help="Set the logging level")

args = parser.parse_args()

logging.basicConfig(stream=sys.stderr, level=args.logLevel,
                        format='%(name)s (%(levelname)s): %(message)s')


logger = logging.getLogger(__name__)
logger.setLevel(args.logLevel)

###############################################################################
# Begin priming the data generation pipeline
###############################################################################

# Get Training and Validation data
train_data = Preprocess(args.image_dir_train)
logger.debug('Completed Preprocess')

train_ds = DataRunner(train_data.set_list,
                      train=True,
                      image_size=args.patch_size)
logger.debug('Completed Data runner')

ds_t = tf.data.Dataset.from_generator(train_ds.get_distributed_datasets, output_types=(
    {
        "anchor": tf.float32,
        "pos_img": tf.float32,
        "neg_img": tf.float32
    }, tf.int64),
    output_shapes=None).batch(args.BATCH_SIZE).repeat()
logger.debug('Completed generator')

if args.image_dir_validation:
    val_data = Preprocess(args.image_dir_validation)

    val_ds = DataRunner(val_data.set_list,
                        train=False,
                        image_size=args.patch_size)

    ds_v = tf.data.Dataset.from_generator(val_ds.get_distributed_datasets, output_types=(
        {
            "anchor": tf.float32,
            "pos_img": tf.float32,
            "neg_img": tf.float32
        }, tf.int64),
        output_shapes=None).batch(args.BATCH_SIZE).repeat()
    validation_steps = val_ds.image_file_list.__len__() / args.BATCH_SIZE

else:
    ds_v = None
    validation_steps = None

# I now have generators for training and validation

###############################################################################
# Build the model
###############################################################################
mirrored_strategy = tf.distribute.MirroredStrategy()
logger.debug('Mirror initialized')

#with mirrored_strategy.scope():
#    m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=128)
#    logger.debug('Model constructed')
#    model = m.compile_model(args.optimizer, args.lr, img_size=args.patch_size)
#    logger.debug('Model compiled')
m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=128)
logger.debug('Model constructed')
model = m.compile_model(args.optimizer, args.lr, img_size=args.patch_size)
logger.debug('Model compiled')

###############################################################################
# Define callbacks
###############################################################################
cb = CallBacks(learning_rate=args.lr, log_dir=args.log_dir, optimizer=args.optimizer)
out_dir = os.path.join(args.log_dir, args.model_name + '_' + args.optimizer + '_' + str(args.lr))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

tf.keras.utils.plot_model(model, to_file=os.path.join(out_dir, 'model.png'), show_shapes=True, show_layer_names=True)
logger.debug('Model image saved')

###############################################################################
# Run the training
###############################################################################
model.fit(ds_t,
          steps_per_epoch=train_ds.image_file_list.__len__() / args.BATCH_SIZE,
          epochs=args.num_epochs,
          callbacks=cb.get_callbacks(),
          validation_data=ds_v,
          validation_steps=validation_steps,
          class_weight=None,
          max_queue_size=1000,
          workers=args.NUM_WORKERS,
          use_multiprocessing=args.use_multiprocessing,
          shuffle=False,
          initial_epoch=0
          )