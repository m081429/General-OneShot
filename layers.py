import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.neg_loss = None
        self.pos_loss = None
        self.loss = None
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = tf.math.reduce_sum(tf.math.square(anchor - positive), axis=-1)
        n_dist = tf.math.reduce_sum(tf.math.square(anchor - negative), axis=-1)
        self.neg_dist = tf.math.reduce_sum(n_dist)
        self.pos_dist = tf.math.reduce_sum(p_dist)
        return tf.math.reduce_sum(tf.math.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        self.loss = loss
        return self.loss, self.neg_dist, self.pos_dist


class LosslessTripletLossLayer(Layer):
    # https://gist.github.com/marcolivierarsenault/a7ef5ab45e1fbb37fbe13b37a0de0257
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.neg_loss = None
        self.pos_loss = None
        self.epsilon = 1e-8
        self.n = 1
        self.beta = 1
        super(LosslessTripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        self.beta = self.n
        # distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        # distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        # Non Linear Values
        # -ln(-x/N+1)
        self.pos_loss = -tf.math.log(-tf.divide(pos_dist, self.beta) + 1 + self.epsilon)
        self.neg_loss = -tf.math.log(-tf.divide((self.n - neg_dist), self.beta) + 1 + self.epsilon)
        # compute loss
        loss = self.neg_loss[0] + self.pos_loss[0]
        return loss

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return tf.dtypes.cast(loss, tf.float32), self.neg_loss, self.pos_loss


def build_triplet_model(patch_size, network, margin=0.2, num_channels=3):
    """
    Define the Keras Model for training
        Input :
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)

    """

    input_shape = (-1, patch_size, patch_size, num_channels)
    # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")

    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)

    # TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin, name='triplet_loss_layer')([encoded_a, encoded_p, encoded_n])

    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)

    # return the model
    return network_train
