import os
import json
from typing import Tuple

import numpy as np
import cv2

from config import DATASET_PATH_BLUE, DATASET_PATH_GREEN, DATASET_PATH_RED, DATASET_PATH_GRAY, DATASET_PATH_COLOUR, \
                   JSON_FILE_PATH, NUMBER_OF_CLASSES


def create_dataset(number_of_images: int, image_wight_range: Tuple[int, int], image_height_range: Tuple[int, int],
                   number_of_classes: int = NUMBER_OF_CLASSES) -> None:
    """
    This function creates dataset of images (red, green, blue, colour, gray)

    :param number_of_images: number of images for one colour in folder.
    :param image_wight_range: the width of the image changes width in the specified range.
    :param image_height_range: the height of the image changes height in the specified range.
    :param number_of_classes: number of image classes.
    """
    # create folders for images
    for p in [DATASET_PATH_BLUE, DATASET_PATH_GREEN, DATASET_PATH_RED, DATASET_PATH_GRAY, DATASET_PATH_COLOUR]:
        os.makedirs(p, exist_ok=True)

    # number of images in one folder
    part_number_of_images = number_of_images // number_of_classes

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
        elif (part_number_of_images * 2) <= i < (part_number_of_images * 3):
            final_image = np.concatenate((image_zeros, image, image_zeros), axis=-1)
            cv2.imwrite(os.path.join(DATASET_PATH_GREEN, 'green_{}.jpg'.format(i)), final_image)
        elif (part_number_of_images * 3) <= i < (part_number_of_images * 4):
            image = np.random.randint(0, 255, (width, height, 1))
            final_image = np.concatenate((image, image, image), axis=-1)
            cv2.imwrite(os.path.join(DATASET_PATH_GRAY, 'gray_{}.jpg'.format(i)), final_image)
        else:
            image_2 = np.random.randint(0, 255, (width, height, 1))
            image_3 = np.random.randint(0, 255, (width, height, 1))
            final_image = np.concatenate((image, image_2, image_3), axis=-1)
            cv2.imwrite(os.path.join(DATASET_PATH_COLOUR, 'colour_{}.jpg'.format(i)), final_image)


def make_data_json(path_for_json: str = JSON_FILE_PATH, proportion_test_images: float = 0.2,
                   number_of_classes: int = NUMBER_OF_CLASSES) -> None:
    """
    This function creates json file with train and test data of images.

    :param path_for_json: this is path where file will save.
    :param proportion_test_images: percentage of test images.
    :param number_of_classes: number of image classes.
    """
    # directory list
    red_catalog = os.listdir(DATASET_PATH_RED)
    green_catalog = os.listdir(DATASET_PATH_GREEN)
    blue_catalog = os.listdir(DATASET_PATH_BLUE)
    gray_catalog = os.listdir(DATASET_PATH_GRAY)
    colour_catalog = os.listdir(DATASET_PATH_COLOUR)

    # number of test images
    len_test_images = int(len(red_catalog + green_catalog + blue_catalog + gray_catalog + colour_catalog) *
                          proportion_test_images)

    # create dictionary
    train_test_image_json = {'train': {}, 'test': {}}

    # create zip object
    path_name_label_zip = zip([DATASET_PATH_RED, DATASET_PATH_GREEN, DATASET_PATH_BLUE, DATASET_PATH_GRAY,
                               DATASET_PATH_COLOUR],
                              [red_catalog, green_catalog, blue_catalog, gray_catalog, colour_catalog],
                              ['red', 'green', 'blue', 'gray', 'colour'])

    # create full dict for json file
    for path_data, name_image, label in path_name_label_zip:
        for n, _ in enumerate(name_image):
            if n < (len_test_images // number_of_classes):
                train_test_image_json['test'][os.path.join(path_data, name_image[n])] = label
            else:
                train_test_image_json['train'][os.path.join(path_data, name_image[n])] = label

    # write json file
    with open(os.path.join(path_for_json), 'w') as f:
        json.dump(train_test_image_json, f, indent=4)


if __name__ == '__main__':
    #create_dataset(number_of_images=5000, image_wight_range=(300, 500), image_height_range=(300, 500))
    make_data_json(proportion_test_images=0.2)
