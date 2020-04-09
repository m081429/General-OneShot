from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import glob
import re
import tensorflow as tf
tf.keras.backend.clear_session()
from callbacks import CallBacks
from model_factory import GetModel
from preprocess import Preprocess, format_example, format_example_tf, update_status, create_triplets_oneshot, create_triplets_oneshot_img
from data_runner import DataRunner
import numpy as np
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

parser.add_argument("-D", "--drop_out",
                    dest="reg_drop_out_per",
                    default=None,type=float,
                    help="Regulrization drop out percent 0-1")

parser.add_argument("--tfrecord_image",
                    dest="tfrecord_image",
                    default="image/encoded",
                    help="Set the logging level")

parser.add_argument("--tfrecord_label",
                    dest="tfrecord_label",
                    default="null",
                    help="Set the logging level")

parser.add_argument("--train_num_layers",
                    dest="train_num_layers",
                    default=False,
                    help="Set the logging level")

parser.add_argument("--prev_checkpoint",
                    dest="prev_checkpoint",
                    default=False,
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
# print(train_data.loss_function)
# print(train_data.filetype)
# print(train_data.loss_function)
# print(train_data.classes)
# print(train_data.tfrecord_image)
# print(train_data.tfrecord_label)

logger.debug('Completed  training dataset Preprocess')

#AUTOTUNE = tf.data.experimental.AUTOTUNE
AUTOTUNE=1000

#Update status to Training for map function in the preprocess
update_status(True)

#If input datatype is tfrecords or images
if train_data.filetype!="tfrecords":
    t_path_ds = tf.data.Dataset.from_tensor_slices(train_data.files)
    t_image_ds = t_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
    t_label_ds = tf.data.Dataset.from_tensor_slices(train_data.labels)
    #t_image_label_ds = tf.data.Dataset.zip((t_image_ds, t_label_ds))
    t_image_label_ds,train_data.min_images=create_triplets_oneshot_img(t_image_ds,t_label_ds)
    # num_img=0    
    # for image, label in t_image_label_ds:
        # print("Image shape: ", image["pos_img"].numpy().shape)
        # print("Image shape: ", image["anchor"].numpy().shape)
        # print("Image shape: ", image["neg_img"].numpy().shape)
        # print("Label: ", label.numpy().shape)
        # print("Label: ", label.numpy())
        # num_img=num_img+1
    # print(num_img,train_data.min_images)    
    # sys.exit(0)
    #train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images).repeat()
else:
    t_path_ds = tf.data.TFRecordDataset(train_data.files)
    t_image_ds = t_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
    #sys.exit(0)
    #min images variables should be update from number  of tfrecords to number of images
    # num_image=0
    # #print(t_image_ds)
    # for image, label in t_image_ds:
        # print(image)
        # print(image.numpy().shape)
        # print(label.numpy())
        # num_image=num_image+1
        # # print(num_image)
        # sys.exit(0)
    #train_data.min_images=num_image
    t_image_label_ds,train_data.min_images=create_triplets_oneshot(t_image_ds)
    #t_image_label_ds=create_triplets_oneshot(t_image_ds)
    
    #t_image_label_ds = tf.data.Dataset.zip(t_image_ds)
    #naresh:adding these additional steps to avoid shuffling on images and shuffle on imagepaths
    #t_image_ds = t_path_ds.shuffle(buffer_size=train_data.min_images).repeat().map(format_example_tf, num_parallel_calls=AUTOTUNE)
    #train_ds = tf.data.Dataset.zip(t_image_ds)
#num_image=0
train_ds_dr = DataRunner(t_image_label_ds)
logger.debug('Completed Data runner')
train_ds = tf.data.Dataset.from_generator(train_ds_dr.get_distributed_datasets, output_types=(
    {
        "anchor": tf.float32,
        "pos_img": tf.float32,
        "neg_img": tf.float32
    }, tf.int64),
    output_shapes=({"anchor": [args.patch_size,args.patch_size,3],"pos_img": [args.patch_size,args.patch_size,3],"neg_img": [args.patch_size,args.patch_size,3]},[3]))
# num_img=0    
# for image, label in train_ds:
    # print("Image shape: ", image["pos_img"].numpy().shape)
    # print("Image shape: ", image["anchor"].numpy().shape)
    # print("Image shape: ", image["neg_img"].numpy().shape)
    # print("Label: ", label.numpy().shape)
    # print("Label: ", label.numpy())
    # num_img=num_img+1
# print(num_img)    
# sys.exit(0)
#train_ds = t_image_label_ds.shuffle(buffer_size=train_data.min_images).repeat()
#train_ds = train_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
train_ds=train_ds.batch(args.BATCH_SIZE).repeat()
training_steps = int(train_data.min_images / args.BATCH_SIZE)
logger.debug('Completed Training dataset')



if args.image_dir_validation:
    # Get Validation data
    #Update status to Testing for map function in the preprocess	
    update_status(False)	 
    validation_data = Preprocess(args.image_dir_validation, args.filetype, args.tfrecord_image, args.tfrecord_label)
    logger.debug('Completed test dataset Preprocess')
	
    if validation_data.filetype!="tfrecords":
        v_path_ds = tf.data.Dataset.from_tensor_slices(validation_data.files)
        v_image_ds = v_path_ds.map(format_example, num_parallel_calls=AUTOTUNE)
        v_label_ds = tf.data.Dataset.from_tensor_slices(validation_data.labels)
        #v_image_label_ds = tf.data.Dataset.zip((v_image_ds, v_label_ds))
        v_image_label_ds,validation_data.min_images=create_triplets_oneshot_img(v_image_ds,v_label_ds)
    else:
        v_path_ds =  tf.data.TFRecordDataset(validation_data.files)
        v_image_ds = v_path_ds.map(format_example_tf, num_parallel_calls=AUTOTUNE)
        #min images variables should be update from number  of tfrecords to number of images		
        #num_image=0
        #for image, label in v_image_ds:
            #num_image=num_image+1
        #print(num_image)
        #sys.exit(0)
        #validation_data.min_images=num_image
        #v_image_label_ds = tf.data.Dataset.zip(v_image_ds)
        v_image_label_ds,validation_data.min_images=create_triplets_oneshot(v_image_ds)
    v_ds_dr = DataRunner(v_image_label_ds)
    logger.debug('Completed Data runner')
    validation_ds = tf.data.Dataset.from_generator(v_ds_dr.get_distributed_datasets, output_types=(
    {
        "anchor": tf.float32,
        "pos_img": tf.float32,
        "neg_img": tf.float32
    }, tf.int64),
    output_shapes=({"anchor": [args.patch_size,args.patch_size,3],"pos_img": [args.patch_size,args.patch_size,3],"neg_img": [args.patch_size,args.patch_size,3]},[3]))
    #validation_ds = v_image_label_ds.shuffle(buffer_size=validation_data.min_images).repeat()
    #validation_ds = validation_ds.batch(args.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    validation_ds=validation_ds.batch(args.BATCH_SIZE).repeat()
    validation_steps = int(validation_data.min_images / args.BATCH_SIZE)
    logger.debug('Completed Validation dataset')

else:
    validation_ds = None
    validation_steps = None

#num_img=0    
# for image, label in train_ds:
    # print("Image shape: ", image["pos_img"].numpy().shape)
    # print("Image shape: ", image["anchor"].numpy().shape)
    # print("Image shape: ", image["neg_img"].numpy().shape)
    # print("Label: ", label.numpy().shape)
    # print("Label: ", label.numpy())
    # num_img=num_img+1
# print(num_img)    
# sys.exit(0)

logger.debug('Mirror initialized')
training_flag=1
if training_flag == 1:
    GPU = True
    if GPU is True:
        # This must be fixed for multi-GPU
        mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with mirrored_strategy.scope():
            m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=128)
            logger.debug('Model constructed')
            model = m.compile_model(args.optimizer, args.lr, img_size=args.patch_size)
            logger.debug('Model compiled')
    else:
        m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=128)
        logger.debug('Model constructed')
        model = m.compile_model(args.optimizer, args.lr, img_size=args.patch_size)
        logger.debug('Model compiled')

    out_dir = os.path.join(args.log_dir, args.model_name + '_' + args.optimizer + '_' + str(args.lr))

    # restore weights if they already exist
    if os.path.exists(os.path.join(out_dir,'my_model.h5')):
        logger.debug('Loading initialized model')
        model = tf.keras.load_model(os.path.join(out_dir,'my_model.h5'))
        logger.debug('Completed loading initialized model')

    ###############################################################################
    # Define callbacks
    ###############################################################################
    cb = CallBacks(learning_rate=args.lr, log_dir=out_dir, optimizer=args.optimizer)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tf.keras.utils.plot_model(model, to_file=os.path.join(out_dir, 'model.png'), show_shapes=True, show_layer_names=True)
    logger.debug('Model image saved')
    #steps_perepoch=train_ds.image_file_list.__len__() / args.BATCH_SIZE

    #featured_img = model.predict(np.ones((1,256,256,3)))
    #print(featured_img)
    #sys.exit(0)
    ###############################################################################
    # Run the training
    ###############################################################################
    model.fit(train_ds,
              steps_per_epoch=training_steps,
              epochs=args.num_epochs,
              callbacks=cb.get_callbacks(),
              validation_data=validation_ds,
              validation_steps=validation_steps,
              class_weight=None,
              max_queue_size=1000,
              workers=args.NUM_WORKERS,
              use_multiprocessing=args.use_multiprocessing,
              shuffle=False,
              initial_epoch=0
              )
    #model.save(os.path.join(out_dir,'my_model.h5'))
    model.save_weights(os.path.join(out_dir,'my_model.h5'))

else:
    out_dir = os.path.join(args.log_dir, args.model_name + '_' + args.optimizer + '_' + str(args.lr))
    m = GetModel(model_name=args.model_name, img_size=args.patch_size, classes=128)
    logger.debug('Model constructed')
    model = m.build_model()
    logger.debug('Model built')
    
    #model = m.compile_model(args.optimizer, args.lr, img_size=args.patch_size)
    #logger.debug('Model compiled')
    latest = tf.train.latest_checkpoint(out_dir)
    print(latest)
    model.load_weights(latest)
    
    for img_data, labels in train_ds:
        #img_data, labels = data
        print(labels.numpy())
        anchor_img, pos_img, neg_img = img_data['anchor'], img_data['pos_img'], img_data['neg_img']
        result = np.asarray(model.predict([anchor_img]))
        print(result)
        sys.exit(0)
        #, pos_img, neg_img])
        result = np.asarray(model.predict([anchor_img, neg_img, pos_img]))
        print(result)
        result = np.asarray(model.predict([anchor_img, neg_img, neg_img]))
        print(result)
        result = np.asarray(model.predict([anchor_img, pos_img, pos_img]))
        print(result)
        sys.exit(0)

