import os
import json
from typing import Tuple

import numpy as np
import cv2

from config import DATASET_PATH_BLUE, DATASET_PATH_GREEN, DATASET_PATH_RED, ALL_PATH


def create_dataset(number_of_images: int, image_wight_range: Tuple[int, int], image_height_range: Tuple[int, int]) \
        -> None:
    """
    This function create dataset of images (red, green, blue)

    :param number_of_images: number of images for one color in folder.
    :param image_wight_range: the wight of the image changes within the specified range.
    :param image_height_range: the height of the image changes within the specified range.
    """
    # create folders for images
    for i in ALL_PATH:
        os.makedirs(i, exist_ok=True)

    # number of images in one folder
    part_number_of_images = number_of_images // 3

    for i in range(number_of_images):
        width = np.random.randint(image_wight_range[0], image_wight_range[1])
        height = np.random.randint(image_height_range[0], image_height_range[1])
        # create and save image
        image = np.random.randint(0, 255, (width, height, 1))
        image_zeros = np.zeros((width, height, 1), np.uint8)
        if i < part_number_of_images:
            final_image = np.concatenate((image_zeros, image_zeros, image), axis=-1)
            cv2.imwrite(os.path.join(DATASET_PATH_RED, 'red_{}.jpg'.format(i)), final_image)
        elif part_number_of_images <= i < (part_number_of_images * 2):
            final_image = np.concatenate((image, image_zeros, image_zeros), axis=-1)
            cv2.imwrite(os.path.join(DATASET_PATH_BLUE, 'blue_{}.jpg'.format(i)), final_image)
        else:
            final_image = np.concatenate((image_zeros, image, image_zeros), axis=-1)
            cv2.imwrite(os.path.join(DATASET_PATH_GREEN, 'green_{}.jpg'.format(i)), final_image)


def make_data_json(proportion_test_images: float) -> None:
    """
    This function create json file with train and test data of images.

    :param proportion_test_images: this parameter is set in the range from 0 to 1.
    """
    # directory list
    red_catalog = os.listdir(DATASET_PATH_RED)
    green_catalog = os.listdir(DATASET_PATH_GREEN)
    blue_catalog = os.listdir(DATASET_PATH_BLUE)

    # number of test images
    len_test_images = int(len(red_catalog + green_catalog + blue_catalog) * proportion_test_images)

    # create dictionary
    train_test_image_json = {'train': {}, 'test': {}}

    # create zip object
    path_name_label_zip = zip([DATASET_PATH_RED, DATASET_PATH_GREEN, DATASET_PATH_BLUE],
                              [red_catalog, green_catalog, blue_catalog],
                              ['red', 'green', 'blue'])

    # create full dict for json file
    for path_data, name_image, label in path_name_label_zip:
        for n in range(len(list(name_image))):
            if n < (len_test_images//3):
                train_test_image_json['test'][os.path.join(path_data, name_image[n])] = label
            else:
                train_test_image_json['train'][os.path.join(path_data, name_image[n])] = label

    # write json file
    with open(os.path.join('data.json'), 'w') as f:
        json.dump(train_test_image_json, f, indent=4)


if __name__ == '__main__':
    create_dataset(number_of_images=300, image_wight_range=(300, 500), image_height_range=(300, 500))
    make_data_json(proportion_test_images=0.2)
