import math
from typing import Tuple, Optional

import keras
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, Activation

from config import INPUT_SHAPE

class Model(keras.models.Model):
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, regularization: Optional[float],
                 activation_type: str, input_name: str) -> None:
        """

        :param input_shape: input shape (height, width, channels).
        :param num_classes: number of classes
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        :param activation_type: Mish
        :param input_name: name of the input tensor
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation_type = activation_type
        self.input_name = input_name

        self.ker_reg = None if regularization is None else keras.regularizers.l2(regularization)

        def custom_activation(x):
            return x * math.tanh(math.log(1 + (math.e ** x)))
        get_custom_objects().update({'custom_activation': Activation(custom_activation)})

        def build(self) -> keras.models.Model:
            """
            Building CNN model for classification task.
            :return: keras.model.Model() object.
            """
            inputs = Input(shape=self.input_shape, name=self.input_name)
            x = BatchNormalization()(inputs)
            # block 1
            x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="same", kernel_regularizer=self.ker_reg)(x)
            x = BatchNormalization()(x)
            x = Activation("custom_activation")(x)
            # block 2
            x = Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same", kernel_regularizer=self.ker_reg)(x)
            x = BatchNormalization()(x)
            x = Activation("custom_activation")(x)
            # block 3
            x = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same", kernel_regularizer=self.ker_reg)(x)
            x = BatchNormalization()(x)
            x = Activation("custom_activation")(x)
            # Dense
            x = GlobalAveragePooling2D()(x)
            x = Dense(self.num_classes, kernel_regularizer=self.ker_reg, activation='softmax')(x)
            return x


r = Model(input_shape=INPUT_SHAPE, num_classes=5, regularization=None, input_name='Privet', activation_type='Mish')





