import tensorflow as tf
from config import NUMBER_OF_CLASSES


class Metric:
    def __init__(self, num_classes: int = NUMBER_OF_CLASSES, is_binary_cross_entropy: bool = False):
        self.num_classes = num_classes
        self.epsilon = 1e-6
        self.is_binary_cross_entropy = is_binary_cross_entropy
        self.__name__ = 'metric'
        if self.is_binary_cross_entropy and self.num_classes != 2:
            msg = 'There should be 2 classes with binary cross entropy, got {}.'.format(self.num_classes)
            raise ValueError(msg)

    def confusion_matrix(self, y_true, y_pred):
        if self.is_binary_cross_entropy:
            y_true = tf.cast(y_true > 0.5, tf.float32)[:, 0]
            y_pred = tf.cast(y_pred > 0.5, tf.float32)[:, 0]
        else:
            y_true = tf.argmax(y_true, 1)
            y_pred = tf.argmax(y_pred, 1)

        matrix = tf.cast(tf.math.confusion_matrix(y_true, y_pred, self.num_classes), tf.float32)
        fp = tf.reduce_sum(matrix, axis=0) - tf.linalg.tensor_diag_part(matrix)
        fn = tf.reduce_sum(matrix, axis=1) - tf.linalg.tensor_diag_part(matrix)
        tp = tf.linalg.tensor_diag_part(matrix)
        return fp, fn, tp
