import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer

# class TripletLossLayer:
    # def __init__(self, alpha, inputs):
        # super().__init__()
        # self.alpha = alpha
        # self.inputs = inputs
        # loss = self.triplet_loss()
        # self.add_loss(loss)
        # #super(TripletLossLayer, self).__init__(**kwargs)
        # return loss
         
    # def triplet_loss(self):
        # anchor, positive, negative = self.inputs
        # p_dist = K.sum(K.square(anchor-positive), axis=-1)
        # n_dist = K.sum(K.square(anchor-negative), axis=-1)
        # return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    # # def call(self, inputs):
        # # loss = self.triplet_loss(inputs)
        # # self.add_loss(loss)
        # # return loss
class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss    
def triplet_loss(y_true, y_pred):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, not used in this function.
    y_pred -- python list containing three objects:
            anchor:   the encodings for the anchor data
            positive: the encodings for the positive data (similar to anchor)
            negative: the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    alpha = 0.2

    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)

    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))

    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    #print('\tNon-Linear distances for pos_dist and neg_dist are {} and {}\n'.format(pos_dist, neg_dist))
    # compute loss
    basic_loss = pos_dist - neg_dist + alpha

    loss = tf.maximum(basic_loss, 0.0)

    return loss


def lossless_triplet_loss(y_true, y_pred, n=3, beta=3, epsilon=1e-8):
    """
    Implementation of the triplet loss function

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    N  --  The number of dimension
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)


    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    #print('\n\tLinear distances for pos_dist and neg_dist are {} and {}'.format(pos_dist,neg_dist ))
    # Non Linear Values

    # -ln(-x/N+1)
    pos_dist = -tf.math.log(-tf.divide(pos_dist, beta) + 1 + epsilon)
    neg_dist = -tf.math.log(-tf.divide((n - neg_dist), beta) + 1 + epsilon)
    #print('\tNon-Linear distances for pos_dist and neg_dist are {} and {}\n'.format(pos_dist,neg_dist))

    # compute loss
    loss = neg_dist + pos_dist

    return loss


