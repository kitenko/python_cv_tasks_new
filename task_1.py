import os
import json
from typing import Tuple

import numpy as np
import cv2


DATASET_PATH = os.path.join('dataset_image')
DATASET_PATH_RED = os.path.join(DATASET_PATH, 'red_image')
DATASET_PATH_BLUE = os.path.join(DATASET_PATH, 'blue_image')
DATASET_PATH_GREEN = os.path.join(DATASET_PATH, 'green_image')
ALL_PATH = [DATASET_PATH_RED, DATASET_PATH_GREEN, DATASET_PATH_BLUE]


def create_dataset(number_of_images: int, image_wight_range: Tuple[int, int], image_height_range: Tuple[int, int],
                   colour: str, create_dataset:bool) -> None:
    """
    This function create dataset of images (red, green, blue)

    :param number_of_images: number of images for one color in folder.
    :param image_wight_range: the wight of the image changes within the specified range.
    :param image_height_range: the height of the image changes within the specified range.
    :param colour: which color of image you want to create
    :param create_dataset: create folders or not for images
    :return: None
    """

    if create_dataset:
        for i in ALL_PATH:
            os.makedirs(i, exist_ok=True)

    if colour == 'red':
        for i in range(number_of_images):
            # random height and width
            width = np.random.randint(image_wight_range[0], image_wight_range[1])
            height = np.random.randint(image_height_range[0], image_height_range[1])
            # create and save image
            image_red = np.random.randint(0, 255, (width, height, 1))
            image_zeros = np.zeros((width, height, 1), np.uint8)
            final_image = np.concatenate((image_zeros, image_zeros, image_red), axis=-1)
            cv2.imwrite(os.path.join(DATASET_PATH_RED, 'red_{}.jpg'.format(i)), final_image)

    elif colour == 'blue':
        for i in range(number_of_images):
            # random height and width
            width = np.random.randint(image_wight_range[0], image_wight_range[1])
            height = np.random.randint(image_height_range[0], image_height_range[1])
            # create and save image
            image_blue = np.random.randint(0, 255, (width, height, 1))
            image_zeros = np.zeros((width, height, 1), np.uint8)
            final_image = np.concatenate((image_blue, image_zeros, image_zeros), axis=-1)
            cv2.imwrite(os.path.join(DATASET_PATH_BLUE, 'blue_{}.jpg'.format(i)), final_image)

    else:
        for i in range(number_of_images):
            # random wight and height
            width = np.random.randint(image_wight_range[0], image_wight_range[1])
            height = np.random.randint(image_height_range[0], image_height_range[1])
            # create and save image
            image_green = np.random.randint(0, 255, (width, height, 1))
            image_zeros = np.zeros((width, height, 1), np.uint8)
            final_image = np.concatenate((image_zeros, image_green, image_zeros), axis=-1)
            cv2.imwrite(os.path.join(DATASET_PATH_GREEN, 'green_{}.jpg'.format(i)), final_image)


def make_data_json(test_image: float) -> None:
    """
    This function create json file with train and test data of images.

    :param test_image: this parameter is set in the range from 0 to 1.
    :return: None
    """

    # directory list
    red_catalog = os.listdir(DATASET_PATH_RED)
    green_catalog = os.listdir(DATASET_PATH_GREEN)
    blue_catalog = os.listdir(DATASET_PATH_BLUE)

    # number of test images
    len_test_red = int(len(red_catalog) * test_image)
    len_test_green = int(len(green_catalog) * test_image)
    len_test_blue = int(len(blue_catalog) * test_image)

    # create dictionary
    train_test_image_json = {'train': {}, 'test': {}}

    # creating a complete dictionary
    for i in range(len(red_catalog)):
        if i < len_test_red:
            train_test_image_json['test'][DATASET_PATH_RED + red_catalog[i]] = 'red'
        else:
            train_test_image_json['train'][DATASET_PATH_RED + red_catalog[i]] = 'red'

    for i in range(len(green_catalog)):
        if i < len_test_green:
            train_test_image_json['test'][DATASET_PATH_GREEN + green_catalog[i]] = 'green'
        else:
            train_test_image_json['train'][DATASET_PATH_GREEN + green_catalog[i]] = 'green'

    for i in range(len(blue_catalog)):
        if i < len_test_blue:
            train_test_image_json['test'][DATASET_PATH_BLUE + blue_catalog[i]] = 'blue'
        else:
            train_test_image_json['train'][DATASET_PATH_BLUE + blue_catalog[i]] = 'blue'

    with open(os.path.join('data.json'), 'w') as f:
        json.dump(train_test_image_json, f, indent=4)


if __name__ == '__main__':

    create_dataset(number_of_images=100, colour='green', image_wight_range=(300, 500), image_height_range=(300, 500),
                   create_dataset=True)
    make_data_json(test_image=0.2)
