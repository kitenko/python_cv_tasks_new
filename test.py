import math
import json
import os
from typing import Tuple

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from config import JSON_FILE_PATH

matplotlib.use('TkAgg')


class DataGenerator:
    def __init__(self, json_path: str, batch_size: int = 16, is_train: bool = True,
                 image_shape: Tuple[int, int, int] = (224, 224, 3)) -> None:
        """
        This function reads parameters, reads json and shuffles the data.

        :param json_path: this is path for json file
        :param batch_size: number of images in one batch
        :param is_train: this is bool value, if is_train = True, then we work with train images, otherwise with test.
        :param image_shape: this is image shape (height, width, channels)
        """
        self.batch_size = batch_size
        self.is_train = is_train
        self.image_shape = image_shape
        self.num_classes = 3

        # read json
        with open(os.path.join(json_path)) as f:  # открыли файл с данными
            train_test_image_json = json.load(f)

        if is_train:
            self.data = train_test_image_json['train']
        else:
            self.data = train_test_image_json['test']

        # shuffle data
        data = list(self.data.items())
        np.random.shuffle(data)
        self.data = data
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """

        :return:
        """
        if self.counter < len(self):
            self.counter += 1
            return self[self.counter - 1]
        else:
            self.counter = 0
            raise StopIteration

    def __len__(self) -> int:
        """
        This function counts numbers of batch

        :return: number_of_batch
        """
        number_of_batch = math.ceil(len(self.data) / self.batch_size)
        return number_of_batch

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Here we return batch (numpy array(batch_size, image_h, image_w, 3). Also we return "onehot encoding", where
        there are batch_size and number of classes.
        :param batch_idx:
        :return:
        """
        data_list = self.data[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)]
        images = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        labels = np.zeros((self.batch_size, self.num_classes))
        for i, (img_path, class_name) in enumerate(data_list):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
            images[i, :, :, :] = resized_image

            if class_name == 'red':
                labels[i, 0] = 1

            elif class_name == 'green':
                labels[i, 1] = 1
            elif class_name == 'blue':
                labels[i, 2] = 1
            else:
                raise ValueError('no label for image')

        return images, labels


def vizualize_data_generator(json_path: str, batch_size: int = 13, is_train: bool = False,
                             image_shape: Tuple[int, int, int] = (224, 224, 3)):
    data_gen = DataGenerator(json_path=json_path, batch_size=batch_size, is_train=is_train, image_shape=image_shape)

    rows_columns_subplot = batch_size
    while math.sqrt(rows_columns_subplot) - int(math.sqrt(rows_columns_subplot)) != 0.0:
        rows_columns_subplot += 1

    rows_columns_subplot = int(math.sqrt(rows_columns_subplot))

    for images_array, labels_array in data_gen:
        plt.figure(figsize=(20, 20))
        for i in range(data_gen.batch_size):
            plt.subplot(rows_columns_subplot, rows_columns_subplot, i+1)
            plt.imshow(images_array[i].astype("uint8"))
            if labels_array[i][0] == 1:
                plt.title("red")
            elif labels_array[i][1] == 1:
                plt.title("green")
            elif labels_array[i][2] == 1:
                plt.title("blue")
            plt.axis("off")
        #plt.show()  у меня получается тоже самое, что если я буду сипользовать plt.show или следующий вывод
        while True:
            if plt.waitforbuttonpress(timeout=- 1):
                break


if __name__ == '__main__':
    #x = DataGenerator(json_path=JSON_FILE_PATH, batch_size=16, is_train=False, image_shape=(224, 224, 3))
    vizualize_data_generator(json_path=JSON_FILE_PATH)
