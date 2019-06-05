import tensorflow as tf
import os

class CallBacks:
    
    def __init__(self, learning_rate=0.01, log_dir=None, optimizer=None):
        super(CallBacks, self).__init__()
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.summary = log_dir + '_' + optimizer + '_' + str(learning_rate)
        self.callbacks = self.get_callbacks()
        

    def _getTB(self):
        return tf.keras.callbacks.TensorBoard(log_dir=self.summary,
                                            histogram_freq=0,
                                            write_graph=False,
                                            update_freq='epoch',
                                            write_images=False)

    def _getCP(self):
        return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.summary,'mymodel.h5'),
                                                save_best_only=True)
    
    def _getES(self):
        return tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    
    def get_callbacks(self):
        return self._getTB(), self._getCP(), self._getES()

