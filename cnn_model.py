import math
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, Activation

from config import INPUT_SHAPE


class My_Mega_Model():
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, regularization: Optional[float],
                 activation_type: str, input_name: str) -> None:
        """

        :param input_shape: input shape (height, width, channels).
        :param num_classes: number of classes
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        :param activation_type: Mish
        :param input_name: name of the input tensor
        """
        self._input_shape = input_shape
        self.num_classes = num_classes
        self.activation_type = activation_type
        self.input_name = input_name

        self.ker_reg = None if regularization is None else keras.regularizers.l2(regularization)

    def custom_activation(self, x):
        return x * tf.math.tanh(tf.math.log1p(1 + (tf.math.exp(1.0) ** x)))

    def build(self) -> tf.keras.Model:
        """
        Building CNN model for classification task.
        :return: keras.model.Model() object.
        """
        inp = Input(shape=self._input_shape, name=self.input_name)
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


#r = Model(input_shape=INPUT_SHAPE, num_classes=5, regularization=None, input_name='Privet', activation_type='Mish')
