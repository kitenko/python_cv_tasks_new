from typing import Tuple
import math
import json
import os
#from math import ceil

import cv2
import numpy as np
import matplotlib.pyplot as plt

JSON_FILE_PATH = os.path.abspath('./')


class DataGenerator:
    def __init__(self, json_path: str, batch_size: int, is_train: bool, image_shape: Tuple[int, int, int]) -> None:
        """
        This function reads parameters, reads json and shuffles the data.

        :param json_path: this is path for json file
        :param batch_size: number of images in one batch
        :param is_train: this is bool value, if is_train = True, then we work with train images, otherwise with test.
        :param image_shape: this is image shape
        """
        self.json_path = json_path
        self.batch_size = batch_size
        self.is_train = is_train
        self.image_shape = image_shape
        self.num_classes = 3

        # read json
        with open(os.path.join(json_path, 'data.json')) as f:  # открыли файл с данными
            train_test_image_json = json.load(f)

        if is_train:
            self.data = train_test_image_json['train']
        else:
            self.data = train_test_image_json['test']

        # shuffle data
        data = list(self.data.items())
        np.random.shuffle(data)
        self.data = dict(data)
        return

    def __iter__(self) -> object:
        return self

    def __next__(self) -> object:
        """

        :return:
        """
        self.counter = 0
        if self.counter < len(self):
            self.counter += 1
            return self[self.counter - 1]
        else:
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
        Here we return batch (numpy array(batch_size, image_h, image_w, 3). Also we return "onehot encoding", where there
        are batch_size and number of classes.
        :param batch_idx:
        :return:
        """
        data_list = list(self.data.items())[(batch_idx * self.batch_size) : ((batch_idx + 1) * self.batch_size)]
        self.images = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        self.labels = np.zeros((self.batch_size, self.num_classes))
        for i, (img_path, class_name) in enumerate(data_list):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(img, (self.image_shape[0], self.image_shape[1]))
            self.images[i, :, :, :] = resized_image

            if class_name == 'red':
                self.labels[i, 0] = 1

            elif class_name == 'green':
                self.labels[i, 1] = 1
            else:
                self.labels[i, 2] = 1

        return self.images, self.labels

    def image_grafic(self):
        """
        This function shows images with labels

        """
        plt.figure(figsize=(10, 10))
        for i in range(self.batch_size):
            image = self.images[i]
            image = image.astype("uint8")
            plt.imshow(image)
            if self.labels[i][0] == 1:
                plt.title("red")
            elif self.labels[i][1] == 1:
                plt.title("green")
            elif self.labels[i][2] == 1:
                plt.title("blue")
            plt.show()

        return


if __name__ == '__main__':
    x = DataGenerator(json_path=JSON_FILE_PATH, batch_size=16, is_train=False, image_shape=(224, 224, 3))
