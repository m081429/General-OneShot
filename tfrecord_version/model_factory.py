import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from losses import TripletLossLayer
from tensorflow.keras import backend as K
#from losses import lossless_triplet_loss as triplet_loss 
class GetModel:

    def __init__(self, model_name=None, img_size=256, classes=1, weights='imagenet', retrain=True, num_layers=None):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.classes = classes
        self.weights = weights
        self.num_layers = num_layers
        self.model, self.preprocess = self.__get_model_and_preprocess(retrain)

    def __get_model_and_preprocess(self, retrain):
        if retrain is True:
            include_top = False
        else:
            include_top = True

        input_tensor = Input(shape=(self.img_size, self.img_size, 3))
        weights = self.weights
        IMG_SHAPE = (self.img_size, self.img_size, 3)

        if self.model_name == 'DenseNet121':
            model = tf.keras.applications.DenseNet121(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.model_name == 'DenseNet169':
            model = tf.keras.applications.DenseNet169(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.model_name == 'DenseNet201':
            model = tf.keras.applications.DenseNet201(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.model_name == 'InceptionResNetV2':
            model = tf.keras.applications.InceptionResNetV2(weights=weights, include_top=include_top,
                                                            input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input(input_tensor)

        elif self.model_name == 'InceptionV3':
            model = tf.keras.applications.InceptionV3(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.inception_v3.preprocess_input(input_tensor)

        elif self.model_name == 'MobileNet':
            model = tf.keras.applications.MobileNet(weights=weights, include_top=include_top,
                                                    input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.mobilenet.preprocess_input(input_tensor)

        elif self.model_name == 'MobileNetV2':
            model = tf.keras.applications.MobileNetV2(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input(input_tensor)

        elif self.model_name == 'NASNetLarge':
            model = tf.keras.applications.NASNetLarge(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.nasnet.preprocess_input(input_tensor)

        elif self.model_name == 'NASNetMobile':
            model = tf.keras.applications.NASNetMobile(weights=weights, include_top=include_top,
                                                       input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.nasnet.preprocess_input(input_tensor)

        elif self.model_name == 'ResNet50':
            model = tf.keras.applications.ResNet50(weights=weights, include_top=include_top,
                                                   input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.resnet50.preprocess_input(input_tensor)

        elif self.model_name == 'VGG16':
            print('Model loaded was VGG16')
            model = tf.keras.applications.VGG16(weights=weights, include_top=include_top,
                                                input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.vgg16.preprocess_input(input_tensor)

        elif self.model_name == 'VGG19':
            model = tf.keras.applications.VGG19(weights=weights, include_top=include_top,
                                                input_tensor=input_tensor, input_shape=IMG_SHAPE)
            preprocess = tf.keras.applications.vgg19.preprocess_input(input_tensor)

        else:
            raise AttributeError("{} not found in available models".format(self.model_name))

        # Add a global average pooling and change the output size to our number of classes
        base_model = model
        x = base_model.output
        x = Flatten()(x)
        #out = Dense(1, activation='sigmoid')(x)
        
        x = Dense(4096, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(x)
        x = Dense(self.classes, activation=None, kernel_regularizer=l2(1e-3),kernel_initializer='he_uniform')(x)
        #Force the encoding to live on the d-dimentional hypershpere
        out = Lambda(lambda x: K.l2_normalize(x,axis=-1))(x)
    
        conv_model = Model(inputs=input_tensor, outputs=out)

        # Now check to see if we are retraining all but the head, or deeper down the stack
        if self.num_layers is not None:
            for layer in base_model.layers[:self.num_layers]:
                layer.trainable = False
            for layer in base_model.layers[self.num_layers:]:
                layer.trainable = True

        return conv_model, preprocess


    def _get_optimizer(self, name, lr=0.001):
        if name == 'Adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
        elif name == 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif name == 'Adamax':
            optimizer = tf.keras.optimizers.Adamax(learning_rate=lr)
        elif name == 'Ftrl':
            optimizer = tf.keras.optimizers.Ftrl(learning_rate=lr)
        elif name == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        elif name == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            raise AttributeError("{} not found in available optimizers".format(name))

        return optimizer
        
    def build_model(self):
        return self.model

    def compile_model(self, optimizer, lr, img_size=256):
        conv_model = self.model

        # Now I need to form my one-shot model structure
        in_dims = [img_size, img_size, 3]

        # Create the 3 inputs
        anchor_in = Input(shape=in_dims, name='anchor')
        pos_in = Input(shape=in_dims, name='pos_img')
        neg_in = Input(shape=in_dims, name='neg_img')

        # Share base network with the 3 inputs
        anchor_out = conv_model(anchor_in)
        pos_out = conv_model(pos_in)
        neg_out = conv_model(neg_in)
        
        #TripletLoss Layer
        margin=0.2
        inputs=[anchor_out,pos_out,neg_out]
        #loss_layer = TripletLossLayer(margin,inputs)
        loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([anchor_out,pos_out,neg_out])
        model = Model(inputs=[anchor_in,pos_in,neg_in],outputs=loss_layer)
        model.compile(loss=None,optimizer=self._get_optimizer(optimizer, lr=lr))
        #y_pred = tf.keras.layers.concatenate([anchor_out, pos_out, neg_out])

        # Define the trainable model
        #model = Model(inputs=[{'anchor': anchor_in,'pos_img': pos_in,'neg_img': neg_in}], outputs=y_pred)
        # model.compile(optimizer=self._get_optimizer(optimizer, lr=lr), loss=triplet_loss, metrics=[
                          # #tf.keras.metrics.AUC(curve='PR', num_thresholds=10, name='PR'),
                          # tf.keras.metrics.AUC( name='AUC'),
                          # tf.keras.metrics.AUC( curve='PR',name='PR'),
                          # tf.keras.metrics.Accuracy(name='accuracy'),
                          # tf.keras.metrics.CategoricalAccuracy(name='CategoricalAccuracy'),
                          # tf.keras.metrics.BinaryAccuracy(name='BinaryAccuracy')
                      # ])

        return model
