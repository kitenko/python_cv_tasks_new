from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, Activation


class GeneratedImagesClassifier:
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, regularization: Optional[float]) -> None:
        """
        This model has three layers Conv2D with BatchNormalization and a "Mish" activation layer.

        :param input_shape: input shape (height, width, channels).
        :param num_classes: number of classes.
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        """
        self.input_shape_image = input_shape
        self.num_classes = num_classes
        self.ker_reg = None if regularization is None else tf.keras.regularizers.l2(regularization)

    @staticmethod
    def custom_activation(x: tf.Tensor) -> tf.Tensor:
        """
        This is the mish activation function.

        :param x:  input tensor.
        :return: output tensor with mish activation.
        """
        return x * tf.math.tanh(tf.math.log1p(1 + (tf.math.exp(1.0) ** x)))

    def build(self) -> tf.keras.Model:
        """
        Building CNN model for classification task.

        :return: keras.model.Model() object.
        """
        inp = Input(shape=self.input_shape_image)
        x = BatchNormalization()(inp)
        # block 1
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="same", kernel_regularizer=self.ker_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(self.custom_activation)(x)
        # block 2
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same", kernel_regularizer=self.ker_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(self.custom_activation)(x)
        # block 3
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same", kernel_regularizer=self.ker_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(self.custom_activation)(x)
        # Dense
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.num_classes, kernel_regularizer=self.ker_reg, activation='softmax')(x)
        model = tf.keras.Model(inputs=inp, outputs=x)
        return model
