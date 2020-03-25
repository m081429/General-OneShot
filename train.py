from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import logging
import os
import sys
import tensorflow as tf
from callbacks import CallBacks
from model_factory import GetModel, build_triplet_model
from preprocess import Preprocess, format_example, format_example_tf, update_status, create_triplets_oneshot
from preprocess import create_triplets_oneshot_img
from data_runner import DataRunner
from steps import write_tb

import re
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
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

parser.add_argument("-c", "--nb_classes",
                    dest='nb_classes',
                    help="How many classes are there to differentiate",
                    default=2, type=int)

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

parser.add_argument("-V", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="DEBUG",
                    help="Set the logging level")

parser.add_argument("-F", "--filetype",
                    dest="filetype",
                    choices=['tfrecords', 'images'],
                    default="images",
                    help="Set the logging level")

parser.add_argument("--tfrecord_image",
                    dest="tfrecord_image",
                    default="image/encoded",
                    help="Set the logging level")

parser.add_argument("--tfrecord_label",
                    dest="tfrecord_label",
                    default="null",
                    help="Set the logging level")

parser.add_argument('-f', "--log_freq",
                    dest="log_freq",
                    default=25,
                    help="Set the logging frequency for saving Tensorboard updates")

args = parser.parse_args()

logging.basicConfig(stream=sys.stderr, level=args.logLevel,
                    format='%(name)s (%(levelname)s): %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(args.logLevel)

###############################################################################
# Begin priming the data generation pipeline
###############################################################################

# Get Training and Validation data
train_data = Preprocess(args.image_dir_train, args.filetype, args.tfrecord_image, args.tfrecord_label)
logger.debug('Completed  training dataset Preprocess')

AUTOTUNE = 1000

# Update status to Training for map function in the preprocess
update_status(True)

# If input datatype is tfrecords or images
if train_data.filetype != "tfrecords":
    t_path_ds = tf.data.Dataset.from_tensor_slices(train_data.files)
    t_image_ds = t_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
    t_label_ds = tf.data.Dataset.from_tensor_slices(train_data.labels)
    t_image_label_ds, train_data.min_images = create_triplets_oneshot_img(t_image_ds, t_label_ds)
else:
    t_path_ds = tf.data.TFRecordDataset(train_data.files)
    t_image_ds = t_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
    t_image_label_ds, train_data.min_images = create_triplets_oneshot(t_image_ds)

train_ds_dr = DataRunner(t_image_label_ds)
logger.debug('Completed Data runner')
train_ds = tf.data.Dataset.from_generator(train_ds_dr.get_distributed_datasets,
                                          output_types=({
                                                            "anchor_img": tf.float32,
                                                            "pos_img": tf.float32,
                                                            "neg_img": tf.float32
                                                        }, tf.int64),
                                          output_shapes=({
                                                             "anchor_img": [args.patch_size, args.patch_size, 3],
                                                             "pos_img": [args.patch_size, args.patch_size, 3],
                                                             "neg_img": [args.patch_size, args.patch_size, 3]
                                                         }, [3]))

train_ds = train_ds.batch(args.BATCH_SIZE)
training_steps = int(train_data.min_images / args.BATCH_SIZE)
logger.debug('Completed Training dataset')

if args.image_dir_validation:
    # Get Validation data
    # Update status to Testing for map function in the preprocess
    update_status(False)
    validation_data = Preprocess(args.image_dir_validation, args.filetype, args.tfrecord_image, args.tfrecord_label)
    logger.debug('Completed test dataset Preprocess')

    if validation_data.filetype != "tfrecords":
        v_path_ds = tf.data.Dataset.from_tensor_slices(validation_data.files)
        v_image_ds = v_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
        v_label_ds = tf.data.Dataset.from_tensor_slices(validation_data.labels)
        v_image_label_ds, validation_data.min_images = create_triplets_oneshot_img(v_image_ds, v_label_ds)
    else:
        v_path_ds = tf.data.TFRecordDataset(validation_data.files)
        v_image_ds = v_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
        v_image_label_ds, validation_data.min_images = create_triplets_oneshot(v_image_ds)
    v_ds_dr = DataRunner(v_image_label_ds)
    logger.debug('Completed Data runner')
    validation_ds = tf.data.Dataset.from_generator(v_ds_dr.get_distributed_datasets,
                                                   output_types=({
                                                                     "anchor_img": tf.float32,
                                                                     "pos_img": tf.float32,
                                                                     "neg_img": tf.float32
                                                                 }, tf.int64),
                                                   output_shapes=({
                                                                      "anchor_img": [args.patch_size, args.patch_size,
                                                                                     3],
                                                                      "pos_img": [args.patch_size, args.patch_size, 3],
                                                                      "neg_img": [args.patch_size, args.patch_size, 3]},
                                                                  [3]))
    validation_ds = validation_ds.batch(args.BATCH_SIZE).repeat()
    validation_steps = int(validation_data.min_images / args.BATCH_SIZE)
    logger.debug('Completed Validation dataset')

else:
    validation_ds = None
    validation_steps = None



# ####################################################################
# Temporary cleaning function
# ####################################################################
out_dir = os.path.join(args.log_dir, args.model_name + '_' + args.optimizer + '_' + str(args.lr))
checkpoint_name = 'training_checkpoints'

overwrite = True
if overwrite is True:
    for root, dirs, files in os.walk(out_dir):
        for file in filter(lambda x: re.match(checkpoint_name, x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))
        for file in filter(lambda x: re.match('checkpoint', x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))
        for file in filter(lambda x: re.match('events', x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))
        for file in filter(lambda x: re.match('ckpt', x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))
        for file in filter(lambda x: re.match('siamese', x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))


###############################################################################
# Define callbacks
###############################################################################
cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


###############################################################################
# Build model
###############################################################################

m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=128)
logger.debug('Model constructed')
model = m.build_model()
logger.debug('Model built')


# Combine triplet Model
siamese_net = build_triplet_model(args.patch_size, model, margin=0.2)
optimizer = m.get_optimizer(args.optimizer)

checkpoint_prefix = os.path.join(out_dir, checkpoint_name)
writer = tf.summary.create_file_writer(out_dir)
checkpoint = tf.train.Checkpoint(model=siamese_net, optimizer=optimizer, step=tf.Variable(1))
manager = tf.train.CheckpointManager(checkpoint, out_dir, max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)  # Restore if necessary

try:
    tf.keras.utils.plot_model(siamese_net, to_file=os.path.join(out_dir, 'model.png'), show_shapes=True,
                              show_layer_names=True)
    logger.debug('Model image saved')
except ImportError:
    print('No pydot available.  Skipping printing')

siamese_net.summary()
###############################################################################
# Run the training
###############################################################################
for epoch in range(args.num_epochs):
    print('Start of epoch %d' % (epoch))
    metadata_file = os.path.join(out_dir, 'metadata.tsv')
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    f = open(metadata_file, 'a')
    # Iterate over the batches of the dataset.
    for step, data in enumerate(train_ds):
        img_data, labels = data
        anchor_img, pos_img, neg_img = img_data['anchor_img'], img_data['pos_img'], img_data['neg_img']

        # write example metadata for embedding visualization
        if step < 1000:
            label = ['anchor', 'pos', 'neg']
            for i in range(0, 3):
                f.write("Step_{}_{}\t{}\n".format(step, label[i], data[1].numpy()[0][i]))

        # Open a GradientTape to record the operations run during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            loss, neg_loss, pos_loss = siamese_net([anchor_img, pos_img, neg_img])

        # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect
        # to the loss.
        grads = tape.gradient(loss, siamese_net.trainable_weights)

        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, siamese_net.trainable_weights))

        print('\rEpoch:{}\tStep:{}\tLoss:{:0.4f}\tneg_dist:{:0.4f}\tpos_dist:{:0.4f}\t'.format(epoch,
                                                                                               step,
                                                                                               loss,
                                                                                               neg_loss[0],
                                                                                               pos_loss[0]), end='')
        if step % args.log_freq == 0 and step > 0:
            checkpoint.step.assign(step)
            write_tb(writer, step, neg_loss[0], pos_loss[0], loss, siamese_net)
            manager.save()
            siamese_net.save_weights(os.path.join(out_dir, 'siamese_net'))

    print('\rEpoch:{}\tStep:{}\tLoss:{:0.4f}\tneg_dist:{:0.4f}\tpos_dist:{:0.4f}\t'.format(epoch,
                                                                                           step,
                                                                                           loss,
                                                                                           neg_loss[0],
                                                                                           pos_loss[0]), end='')
    print('')  # Create a newline
    manager.save()
manager.save()