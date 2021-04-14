import math
from typing import Tuple
import json
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from math import ceil


class DataGenerator:
    def __init__(self, json_path: str, batch_size: int, is_train: bool, image_shape: Tuple[int, int, int]) -> None:

        self.json_path = json_path
        self.batch_size = batch_size
        self.is_train = is_train
        self.image_shape = image_shape
        self.num_classes = 3
        print(self.image_shape[1])

        # read json
        with open('datasetdata.json') as f:  # открыли файл с данными
            dicti = json.load(f)

        if is_train:
            self.data = dicti['train']
        else:
            self.data = dicti['test']

        # shuffle data
        l = list(self.data.items())
        random.shuffle(l)
        self.data = dict(l)
        print(len(self.data))


    def __iter__(self):
        return self

    def __next__(self):

        self.counter = 0
        if self.counter < len(self):
            self.counter += 1
            return self[self.counter - 1]
        else:
            raise StopIteration

    def __len__(self):

        self.number_of_batch = math.ceil(len(self.data) / self.batch_size)
        return self.number_of_batch


    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        self.exit_batch = np.zeros((16, 224, 224, 3))
        data = self.data
        data_list = sorted(data.items())[(batch_idx * self.batch_size) : ((batch_idx + 1) * self.batch_size)]
        data_list = random.sample(data_list, len(data_list))
        self.images = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        self.labels = np.zeros((self.batch_size, self.num_classes))
        for i, (img_path, class_name) in enumerate(data_list):
            img = cv2.imread(img_path)
            self.resized_image = cv2.resize(img, (self.image_shape[0], self.image_shape[1]))
            self.images[i, :, :, :] = self.resized_image

            if class_name == 'red':
                self.labels[i, 0] = 1

            elif class_name == 'green':
                self.labels[i, 1] = 1
            else:
                self.labels[i, 2] = 1

        return self.images, self.labels


    def image_grafic(self):
        plt.figure(figsize=(10, 10))
        for i in range(self.batch_size):
            image = self.images[i]
            image = image.astype("uint8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            if self.labels[i][0] == 1:
                plt.title("red")
            elif self.labels[i][1] == 1:
                plt.title("green")
            elif self.labels[i][2] == 1:
                plt.title("blue")

            plt.show()







x = DataGenerator(json_path='/home/andre/pycharm/python_cv_tasks/task_1', batch_size=16, is_train=False, image_shape=(224, 224, 3))

x.__getitem__(1)
x.image_grafic()