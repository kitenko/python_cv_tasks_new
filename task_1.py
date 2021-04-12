##
import os
import numpy as np
import cv2
import random
import json

##
# создание каталогов

"""os.mkdir("dataset")
os.mkdir("dataset/red")
os.mkdir("dataset/blue")
os.mkdir("dataset/green")"""

##
path_dataset = "dataset"
path_red = "dataset/red"
path_blue = "dataset/blue"
path_green = "dataset/green"
path_green = os.path.join(path_dataset, 'green')

arr = np.random.randint()

def create_dataset(path = path_dataset, number_of_images = 100, width_image = (100, 300), height_image = (100, 300), colour = ''):

    def create_blank(width, height, rgb_color=(0, 0, 0)):
        #Create new image(numpy array) filled with certain color in RGB
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)

        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        image[:] = color

        return image

    if colour == 'red':
        os.chdir(path)
        for i in range(number_of_images):
            red = (random.randint(0, 255), 0, 0)
            image = create_blank(random.randint(width_image[0], width_image[1]),
                                 random.randint(height_image[0], height_image[1]), rgb_color=red)
            cv2.imwrite('red_'+str(i)+'.jpg', image)

    elif colour == 'green':
        os.chdir(path)
        for i in range(number_of_images):
            green = (0, random.randint(0, 255), 0)
            image = create_blank(random.randint(width_image[0], width_image[1]),
                                 random.randint(height_image[0], height_image[1]), rgb_color=green)

            cv2.imwrite('green_' + str(i) + '.jpg', image)

    else:
        os.chdir(path)
        for i in range(number_of_images):
            blue = (0, 0, random.randint(0, 255))
            image = create_blank(random.randint(width_image[0], width_image[1]),
                                 random.randint(height_image[0], height_image[1]), rgb_color=blue)

            cv2.imwrite('blue_' + str(i) + '.jpg', image)
    return

# create_dataset(path = path_green, number_of_images=100, colour= 'green')




def make_data_json(path_dataset = path_dataset, path_json = path_dataset, test_image = 10):

    # просмотр каталогов
    data_catalog = os.listdir(path_dataset)
    red_catalog = os.listdir(path_dataset + "/red")
    green_catalog = os.listdir(path_dataset + "/green")
    blue_catalog = os.listdir(path_dataset + "/blue")

    # количество тестовых изображений
    len_test_red = int(((len(red_catalog)/100) * test_image))
    len_test_green = int(((len(green_catalog)/100) * test_image))
    len_test_blue = int(((len(blue_catalog)/100) * test_image))

    fail_slov = {"train" : {

            },
            "test" : {

            }
                 }

    # перемешевание каталогов перед распределением
    random.shuffle(red_catalog)
    random.shuffle(green_catalog)
    random.shuffle(blue_catalog)

    # создание полного словаря

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

make_data_json(test_image=20)

##

#загрузить из json
with open('datasetdata.json', 'r', encoding='utf-8') as fh: #открываем файл на чтение
    data = json.load(fh) #загружаем из файла данные в словарь data

##

#print(data)
keys = data.keys()
values = data.values()
random.shuffle(values)
a_shuffled = dict(zip(keys, values))
print(a_shuffled)


##

if __name__ == '__main__':
    # create_dataset()
    # create_dataset()
    arr = np.random.randint
    pass
