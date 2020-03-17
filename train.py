from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import logging
import os
import sys
import tensorflow as tf
from callbacks import CallBacks
from model_factory import GetModel
from preprocess import Preprocess, format_example, format_example_tf, update_status, create_triplets_oneshot
from preprocess import create_triplets_oneshot_img
from data_runner import DataRunner
from losses import triplet_loss as loss_fn

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

parser.add_argument("--use-multiprocessing",
                    help="Whether or not to use multiprocessing",
                    const=True, default=False, nargs='?',
                    type=bool)

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

m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=128)
logger.debug('Model constructed')
model = m.build_model()
logger.debug('Model built')

out_dir = os.path.join(args.log_dir, args.model_name + '_' + args.optimizer + '_' + str(args.lr))

###############################################################################
# Define callbacks
###############################################################################
cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

try:
    tensorflow.keras.utils.plot_model(model, to_file=os.path.join(out_dir, 'model.png'), show_shapes=True,
                              show_layer_names=True)
    logger.debug('Model image saved')
except ImportError:
    print('No pydot available.  Skipping printing')

###############################################################################
# Run the training
###############################################################################
optimizer = m.get_optimizer(args.optimizer)

for epoch in range(args.num_epochs):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, data in enumerate(train_ds):
        img_data, labels = data
        anchor_img, pos_img, neg_img = img_data['anchor_img'], img_data['pos_img'], img_data['neg_img']
        # Open a GradientTape to record the operations run during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            # Get embeddings for each image type
            z0 = model(anchor_img)
            z1 = model(pos_img)
            z2 = model(neg_img)

            # Compute the loss value for this minibatch.
            # Returned both so I can print independent of each other
            # This function maximizes the distance between the anchor and negative case while minimizing the
            # difference between the anchor and positive
            neg_dist, pos_dist = loss_fn(anchor=z0,
                                         positive=z1,
                                         negative=z2)

            # Ensure there is always a non-zero overall distance
            total_dist = tf.math.maximum(neg_dist + pos_dist + 1e-8, 1e-8)
            print('\rStep: {}\tNeg_Loss: {}\tPos_Loss: {}\tOverall: {}'.format(step, neg_dist, pos_dist, total_dist),
                  end='')

        # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect
        # to the loss.
        grads = tape.gradient(total_dist, model.trainable_weights)

        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches. #TODO: Create summary file writer so I can write to tensorboard
        # if step % 10 == 0:
        # tf.summary.scalar('Negative_distance', neg_dist, step=step)
        # tf.summary.scalar('Positive_distance', pos_dist, step=step)
        # tf.summary.scalar('Total_distance', total_dist, step=step)
        print('\rStep: {}\tNeg_Loss: {}\tPos_Loss: {}\t'.format(step, neg_dist, pos_dist), end='')

    print('')  # Create a newline

model.save(os.path.join(out_dir, 'my_model.h5'))
