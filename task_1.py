import os
import numpy as np
import cv2
import random
import json


"""Я пытлася реализовать чтобы пути папок были доступтны везде, но global так и не помог. Из-за этого мне необходимо
прописовать пути отдельно в виде констант, как ты и говорил. Но есть ли возможность сделать, что бы мои переменные были
 достурты из вложанной функции? """


path_dataset = "dataset"
path_red = path_dataset + "/red"
path_blue = path_dataset + "/blue"
path_green = path_dataset + "/green"


def create_dataset(path: str, number_of_images=100, width_image=(100, 300), height_image=(100, 300), colour=''):

    def create_directory(path: str):

        # оставил для примера

        #global path_red
        #global path_blue
        #global path_green

        path_red = path + "/red"
        path_blue = path + "/blue"
        path_green = path + "/green"

        os.makedirs(path_red, exist_ok=True)
        os.makedirs(path_blue, exist_ok=True)
        os.makedirs(path_green, exist_ok=True)

        return

    create_directory(path)


    if colour == 'red':
        os.chdir(path_red)
        for i in range(number_of_images):

            # random height and width
            width = np.random.randint(width_image[0], width_image[1])
            height = np.random.randint(height_image[0], height_image[1])

            # create and save image
            image_red = np.random.randint(0, 255, (width, height, 1))
            image_zeros = np.zeros((width, height, 1), np.uint8)
            final_image = np.concatenate((image_zeros, image_zeros, image_red), axis=-1)
            cv2.imwrite('red_'+str(i)+'.jpg', final_image)

    elif colour == 'blue':
        os.chdir(path_blue)
        for i in range(number_of_images):

            # random height and width
            width = np.random.randint(width_image[0], width_image[1])
            height = np.random.randint(height_image[0], height_image[1])

            # create and save image
            image_blue = np.random.randint(0, 255, (width, height, 1))
            image_zeros = np.zeros((width, height, 1), np.uint8)
            final_image = np.concatenate((image_blue, image_zeros, image_zeros), axis=-1)
            cv2.imwrite('blue_' + str(i) + '.jpg', final_image)

    else:
        os.chdir(path_green)
        for i in range(number_of_images):

            # random wight and height
            width = np.random.randint(width_image[0], width_image[1])
            height = np.random.randint(height_image[0], height_image[1])

            # create and save image
            image_green = np.random.randint(0, 255, (width, height, 1))
            image_zeros = np.zeros((width, height, 1), np.uint8)
            final_image = np.concatenate((image_zeros, image_green, image_zeros), axis=-1)
            cv2.imwrite('green_' + str(i) + '.jpg', final_image)


#create_dataset(path="dataset", number_of_images=100, colour='green', width_image=(300, 500), height_image=(300, 500))


def make_data_json(path_dataset: str, path_json=path_dataset, test_image=10):

    # directory list
    red_catalog = os.listdir(path_red)
    green_catalog = os.listdir(path_green)
    blue_catalog = os.listdir(path_blue)


    # number of test images
    len_test_red = int(((len(red_catalog)/100) * test_image))
    len_test_green = int(((len(green_catalog)/100) * test_image))
    len_test_blue = int(((len(blue_catalog)/100) * test_image))

    # create dictionary
    fail_slov = {"train" : {

            },
            "test" : {

            }
                 }

    # shuffling directories before distribution
    random.shuffle(red_catalog)
    random.shuffle(green_catalog)
    random.shuffle(blue_catalog)

        # creating a complete dictionary
    for i in range(len(red_catalog)):
        if i < len_test_red:
            fail_slov["test"][path_red + red_catalog[i]] = "red"
        else:
            fail_slov["train"][path_red + red_catalog[i]] = "red"

    for i in range(len(green_catalog)):
        if i < len_test_green:
            fail_slov["test"][path_green + green_catalog[i]] = "green"
        else:
            fail_slov["train"][path_green + green_catalog[i]] = "green"

    for i in range(len(blue_catalog)):
        if i < len_test_blue:
            fail_slov["test"][path_blue + blue_catalog[i]] = "blue"
        else:
            fail_slov["train"][path_blue + blue_catalog[i]] = "blue"

    with open(path_json + 'data.json', 'w', encoding='utf-8') as f:
        json.dump(fail_slov, f, ensure_ascii=False, indent=4)

#make_data_json(path_dataset, test_image=20)